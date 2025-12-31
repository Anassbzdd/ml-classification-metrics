from sklearn.metrics import confusion_matrix, accuracy_score, recall_score , precision_score , f1_score , roc_curve

y_true = [1,0,1,0,1,0,1,0,0,1]
y_pred = [1,0,1,0,1,0,1,0,0,0]

cm = confusion_matrix(y_true , y_pred)
tp , fp , tn , fn = cm.ravel()

print("Confusion Matrix:\n", cm)
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

acc = accuracy_score(y_true,y_pred)
rec = recall_score(y_true,y_pred)
pre = precision_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
fpr = fp / (fp + tn)

print(f"\nAccuracy: {acc:.2f}")
print(f"Recall: {rec:.2f}")
print(f"Precision: {pre:.2f}")
print(f"F1: {rec:.2f}")
print(f"False Positive Rate: {fpr:.2f}")
