import os
import logging
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

def features_to_text(feature_vector: np.ndarray, feature_names: list, top_k: int = 10) -> str:
    """
    Converts scaled numerical features of a network flow into descriptive English sentences.
    """
    ranked_idx = np.argsort(np.abs(feature_vector))[::-1][:top_k]
    parts = []
    for i in ranked_idx:
        name = feature_names[i].strip()
        val  = feature_vector[i]
        if "pkt" in name.lower() or "bytes" in name.lower():
            parts.append(f"{name} is {'high' if val > 0.5 else 'low'} ({val:.2f})")
        elif "flag" in name.lower():
            parts.append(f"{name} count is {abs(val):.1f}")
        elif "duration" in name.lower():
            parts.append(f"flow duration is {'long' if val > 0 else 'short'}")
        elif "rate" in name.lower() or "pkts/s" in name.lower():
            parts.append(f"traffic rate is {'elevated' if val > 0.5 else 'normal'} ({val:.2f})")
        else:
            parts.append(f"{name} = {val:.2f}")
    return "Network flow characteristics: " + "; ".join(parts) + "."

def build_fallback_explanation(prediction: int, probability: float,
                               feature_vector: np.ndarray, feature_names: list,
                               attack_label: str = None) -> str:
    """
    Rule-based fallback explanation generator if the OpenAI API call fails.
    """
    label     = "ANOMALY (Attack)" if prediction == 1 else "BENIGN"
    conf      = probability if prediction == 1 else 1 - probability
    flow_text = features_to_text(feature_vector, feature_names)

    explanation = (
        f"[PREDICTION]: {label} | Confidence: {conf*100:.1f}%\n"
        f"[FLOW DESCRIPTION]: {flow_text}\n"
    )
    if prediction == 1:
        explanation += (
            f"[REASON]: This network flow exhibits abnormal behavior consistent with "
            f"{attack_label or 'malicious'} IoT traffic (e.g., DoS flooding, scanning). "
            f"The GAN-augmented classifier flagged this with high confidence (Fallback Rule-based Explanation).\n"
            f"[ATTACK TYPE]: {attack_label or 'Unknown Attack'}\n"
            f"[RECOMMENDATION]: Isolate the IoT device. Block the source IP on the firewall. "
            f"Review traffic rate limiting policies."
        )
    else:
        explanation += (
            f"[REASON]: Flow characteristics match normal IoT device communication. "
            f"Packet sizes, rates, and flag patterns are within expected thresholds (Fallback Rule-based Explanation).\n"
            f"[RECOMMENDATION]: No action required. Continue monitoring."
        )
    return explanation

class LlmAnomalyExplainer:
    def __init__(self, config):
        self.config = config
        self.api_key = getattr(config, "OPENAI_API_KEY", None)
        self.model = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not defined in configuration. Fallback to rule-based explanations will be used.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Explainer initialized successfully using model: %s", self.model)
            except Exception as e:
                logger.error("Failed to initialize OpenAI client: %s", e)
                self.client = None

    def explain_flow(self, prediction: int, probability: float,
                     feature_vector: np.ndarray, feature_names: list,
                     attack_label: str = None) -> str:
        """
        Sends the network flow metrics to OpenAI to generate a rich, natural language explanation.
        """
        if self.client is None:
            return build_fallback_explanation(prediction, probability, feature_vector, feature_names, attack_label)

        # Generate readable feature summary
        flow_text = features_to_text(feature_vector, feature_names)
        label_str = "ANOMALY (Attack)" if prediction == 1 else "BENIGN"
        conf_pct = f"{probability*100:.1f}%" if prediction == 1 else f"{(1 - probability)*100:.1f}%"

        prompt = f"""You are a Cyber Security AI Assistant analyzing network traffic anomalies for IoT devices.
An anomaly detection system (trained using a GAN-augmented classifier) has predicted a flow as follows:

--- ALERT CONTEXT ---
Prediction: {label_str}
Detection Confidence: {conf_pct}
Target Attack Category in Metadata: {attack_label or 'N/A'} (Botnet vs. Malware)

--- NETWORK METRICS (Standardized feature vector where 0 is the benign average) ---
{flow_text}

--- YOUR TASK ---
Write a structured security explanation for this alert. Follow this EXACT format (incorporate your answers directly beneath the headers):

[PREDICTION]: {label_str} | Confidence: {conf_pct}
[FLOW DESCRIPTION]: {flow_text}
[REASON]: (Interpret the numeric features in plain English and explain how they relate to Botnet or Malware traffic patterns, e.g., DDoS flooding, slow-rate exhaustion, or anomalous protocol flag counts.)
[ATTACK TYPE]: {attack_label or 'N/A'}
[RECOMMENDATION]: (Provide concrete, actionable steps to secure the IoT device and mitigate the threat.)

Keep the explanation concise, professional, and strictly formatted as requested.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert security analyst specialized in IoT intrusion detection systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            logger.error("Failed to generate OpenAI explanation: %s. Falling back to rule-based explanation.", e)
            return build_fallback_explanation(prediction, probability, feature_vector, feature_names, attack_label)

    def generate_explanations(self, X_test: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray, att_labels: np.ndarray,
                               feature_names: list, n_samples: int = 10) -> list:
        """
        Generates explanations for a random sample of the test set predictions.
        """
        explanations = []
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        logger.info("Generating OpenAI explanations for %d test samples...", len(indices))
        for idx in indices:
            pred_class = int(y_pred[idx])
            pred_prob  = float(y_proba[idx])
            att_type   = str(att_labels[idx]) if att_labels is not None else "N/A"

            exp = self.explain_flow(
                prediction=pred_class,
                probability=pred_prob,
                feature_vector=X_test[idx],
                feature_names=feature_names,
                attack_label=att_type
            )

            explanations.append({
                "sample_idx":   int(idx),
                "prediction":   "Anomaly" if y_pred[idx] == 1 else "Benign",
                "confidence":   f"{pred_prob*100:.1f}%" if y_pred[idx] == 1 else f"{(1 - pred_prob)*100:.1f}%",
                "attack_type":  att_type,
                "explanation":  exp,
            })
        return explanations
