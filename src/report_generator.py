def generate_report(group_acc, bias_score):
    report = "=== AI Ethics Skill-Profiler ===\n\n"
    report += "Gruppen-Genauigkeiten:\n"

    for group, acc in group_acc.items():
        report += f"  - {group}: {acc:.2f}\n"

    report += f"\nBias Score: {bias_score:.2f}\n"

    if bias_score > 0.10:
        report += "⚠️  Hinweis: Das Modell zeigt deutliche Unterschiede zwischen Gruppen.\n"
    else:
        report += "✔️  Das Modell wirkt fair.\n"

    return report
