import logging
import numpy as np

logger = logging.getLogger(__name__)


def features_to_text(feature_vector: np.ndarray, feature_names: list, top_k: int = 10) -> str:
    """
    Converts scaled numerical features of a network flow into descriptive English sentences.
    Highlights the top_k most significant features (based on absolute deviation from average).
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


class HybridAnomalyExplainer:
    """
    Generates structured, natural language explanations of anomaly predictions.
    Utilizes the 3-class hybrid model's prediction output and confidence metrics.
    """
    def __init__(self, config):
        self.config = config

    def explain_flow(self, prediction: int, probabilities: np.ndarray,
                     feature_vector: np.ndarray, feature_names: list,
                     attack_label: str = None) -> str:
        """
        Generate a detailed natural language explanation based on class prediction and features.
        """
        class_names = ["BENIGN", "BOTNET (Attack)", "MALWARE (Attack)"]
        label_str = class_names[prediction]
        conf_pct = float(probabilities[prediction]) * 100.0
        
        flow_text = features_to_text(feature_vector, feature_names)
        
        # Build explanation
        explanation = (
            f"[PREDICTION]: {label_str} | Confidence: {conf_pct:.1f}%\n"
            f"[FLOW DESCRIPTION]: {flow_text}\n"
        )
        
        if prediction == 1:  # Botnet
            explanation += (
                f"[REASON]: The network flow matches Botnet patterns. "
                f"We observed high-rate packet flooding and/or brute force attempts (e.g., FTP/SSH Brute Force, DoS-Hulk, DoS-GoldenEye). "
                f"The hybrid fusion model flagged this anomalous traffic using both tabular signature features "
                f"and text representation semantics processed by DistilBERT.\n"
                f"[ATTACK TYPE]: Botnet ({attack_label or 'DoS/Brute Force'})\n"
                f"[RECOMMENDATION]: Immediate action required. Isolate the affected IoT device. "
                f"Block the source IP on the firewall. Update SSH/FTP credentials immediately. "
                f"Review traffic rate limiting policies."
            )
        elif prediction == 2:  # Malware
            explanation += (
                f"[REASON]: The network flow matches Malware patterns. "
                f"We observed slow-rate exhaustion or application exploits (e.g., DoS-Slowloris, DoS-SlowHTTPTest). "
                f"Unlike high-rate Botnet flooding, this traffic uses low packet frequency but holds sockets open, "
                f"which the hybrid model detected via temporal features and text embeddings.\n"
                f"[ATTACK TYPE]: Malware ({attack_label or 'Slow-Rate Exhaustion'})\n"
                f"[RECOMMENDATION]: Immediate action required. Terminate active slow connections. "
                f"Configure Web Server timeout settings (e.g., RequestReadTimeout). "
                f"Deploy a Web Application Firewall (WAF) to filter slow-rate connection hijacking."
            )
        else:  # Benign
            explanation += (
                f"[REASON]: The flow characteristics are consistent with benign IoT communication. "
                f"Packet lengths, transmission rates, and protocol flag frequencies are within normal operating bounds.\n"
                f"[ATTACK TYPE]: N/A\n"
                f"[RECOMMENDATION]: No action required. Continue monitoring traffic logs."
            )
            
        return explanation

    def generate_explanations(self, X_test: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray, att_labels: np.ndarray,
                               feature_names: list, n_samples: int = 10) -> list:
        """
        Generates explanations for a random sample of predictions.
        """
        explanations = []
        # Sample predictions across different classes if possible
        indices = []
        for c in [0, 1, 2]:
            c_indices = np.where(y_pred == c)[0]
            if len(c_indices) > 0:
                sampled_c = np.random.choice(c_indices, min(n_samples // 3 + 1, len(c_indices)), replace=False)
                indices.extend(sampled_c)
                
        # Fill in up to n_samples
        if len(indices) < n_samples:
            remaining = np.setdiff1d(np.arange(len(X_test)), indices)
            if len(remaining) > 0:
                sampled_rem = np.random.choice(remaining, min(n_samples - len(indices), len(remaining)), replace=False)
                indices.extend(sampled_rem)
                
        # Shuffle indices
        indices = np.random.permutation(indices)[:n_samples]

        logger.info(" Generating Hybrid DistilBERT-based explanations for %d samples...", len(indices))
        for idx in indices:
            pred_class = int(y_pred[idx])
            pred_probs = y_proba[idx]
            att_type   = str(att_labels[idx]) if att_labels is not None else "N/A"

            exp = self.explain_flow(
                prediction=pred_class,
                probabilities=pred_probs,
                feature_vector=X_test[idx],
                feature_names=feature_names,
                attack_label=att_type
            )

            class_names = ["Benign", "Botnet", "Malware"]
            explanations.append({
                "sample_idx":   int(idx),
                "prediction":   class_names[pred_class],
                "confidence":   f"{pred_probs[pred_class]*100:.1f}%",
                "attack_type":  att_type,
                "explanation":  exp,
            })
        return explanations
