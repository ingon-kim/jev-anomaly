from eval_anomaly import auroc, binary_metrics, best_f1
assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
assert auroc([0, 1], [0.5, 0.5]) == 0.5  # ties
assert auroc([0, 0, 1], [0.1, 0.2, 0.9]) == 1.0
m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
assert (m["accuracy"], m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5, 0.5)
assert best_f1([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])["threshold"] == 0.4
print("ok")
