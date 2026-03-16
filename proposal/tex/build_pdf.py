#!/usr/bin/env python3
"""Build proposal.pdf from proposal content (no LaTeX required)."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT

def _add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(
        "proposal.pdf",
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="ProposalTitle",
        parent=styles["Heading1"],
        fontSize=14,
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        name="Bullet",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=28,
        firstLineIndent=-8,
        spaceAfter=2,
    )
    subbullet_style = ParagraphStyle(
        name="SubBullet",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=44,
        firstLineIndent=-8,
        spaceAfter=2,
    )

    story = []

    story.append(Paragraph(
        "Stroke Prediction with Class Imbalance: Model Comparison and Handling Strategies",
        title_style,
    ))
    story.append(Paragraph("CSCI 567 Project Proposal", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Requirements", heading_style))

    story.append(Paragraph("What research question are you trying to answer?", heading_style))
    story.append(Paragraph(
        "• <b>Primary question:</b> Which features and ML algorithms best predict stroke occurrence in this dataset?",
        bullet_style,
    ))
    story.append(Paragraph(
        "• <b>Secondary question:</b> Can we improve prediction by addressing class imbalance?",
        bullet_style,
    ))
    story.append(Paragraph(
        "  • The dataset is highly imbalanced: approximately 95% of records are no-stroke and 5% are stroke.",
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • How effective are existing methods for handling class imbalance (e.g., resampling, class weights)?",
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • Can we propose or adapt methods to better deal with this imbalance?",
        subbullet_style,
    ))

    story.append(Paragraph("Why is this question interesting to you?", heading_style))
    story.append(Paragraph(
        "• <b>Interest:</b> We are drawn to exploring class imbalance and the predictive power of different models on a dataset in the medical field.",
        bullet_style,
    ))
    story.append(Paragraph(
        "• <b>Motivation:</b> Real world data is messy and often has extreme cases of class imbalance, which could result in very high accuracy scores but low F1. Being able to gracefully handle class imbalance to produce accurate predictions is especially important in high-stakes situations such as predicting whether a patient is likely to have a stroke. False negatives could prevent patients from seeking preventative care or making the necessary lifestyle changes to minimize the risk of stroke.",
        bullet_style,
    ))

    story.append(Paragraph("What kind of data are you collecting or what datasets will you use?", heading_style))
    story.append(Paragraph(
        '• We use the <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data" color="blue">Kaggle Stroke Prediction Dataset</a>. The data has 10 features: gender, age, hypertension, heart disease, marriage status, work type, residence type, avg. glucose level, BMI, and smoking status—used to predict stroke (target provided in the dataset). Seven are categorical and three are numerical.',
        bullet_style,
    ))
    story.append(Paragraph(
        "• The data passes basic sanity checks, which indicate that it is likely real rather than synthetically generated (e.g., no young children with married/smoking/non-child work-type entries). There are some missing BMI values, consistent with physically collected records.",
        bullet_style,
    ))
    story.append(Paragraph(
        '• Kaggle states the data comes from patient records. The dataset was created by <a href="https://www.kaggle.com/fedesoriano" color="blue">fedesoriano</a>, a Kaggle Datasets Grandmaster, which we take as an indication of reliability and legitimacy.',
        bullet_style,
    ))

    story.append(Paragraph("What algorithms will you try?", heading_style))
    story.append(Paragraph(
        '• We will try: Random Forest; neural networks with regularization (<a href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html" color="blue">scikit-learn MLPClassifier</a>); Logistic regression; SVM; <a href="https://xgboost.readthedocs.io/en/stable/" color="blue">XGBoost</a>.',
        bullet_style,
    ))

    story.append(PageBreak())
    story.append(Paragraph("What experiments and analysis will you run?", heading_style))
    story.append(Paragraph(
        "• We will evaluate predictive performance (e.g., accuracy, F1, recall) for binary stroke vs no-stroke prediction.",
        bullet_style,
    ))
    story.append(Paragraph("• <b>Planned analyses:</b>", bullet_style))
    story.append(Paragraph(
        '  • <b>Class imbalance:</b> Compare resampling (oversampling/undersampling), class weights, etc., vs baseline; report recall and F1. We will refer to and build on <a href="https://hemostasistoday.com/insight/favour-kpokpe-32544" color="blue">this analysis of data imbalance</a>.',
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • <b>Feature importance:</b> XGBoost, ablation; optionally l_0 regularization.",
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • <b>Model comparison:</b> Compare all models via hold-out or cross-validation (accuracy, F1, recall).",
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • <b>Overfitting:</b> Use train / validation / test split and monitor generalization.",
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • <b>Categorical and numerical features:</b> One-hot encoding; concatenate features (e.g., age, bmi, is_Male, is_Female, …).",
        subbullet_style,
    ))
    story.append(Paragraph(
        "  • <b>Missing BMI:</b> Compare dropping missing values vs imputation (e.g., from observed data).",
        subbullet_style,
    ))

    story.append(Paragraph("What do you plan to finish by the pre-final report and check-in? (Check in April 20th)", heading_style))
    story.append(Paragraph(
        "• <b>Experiments:</b> By the week of April 20th, we will have completed experiments and analysis for at least two of our algorithms (Random Forest, neural networks, Logistic regression, SVM, XGBoost) and made substantial progress on the third, with time set aside for debugging and sound experimental design.",
        bullet_style,
    ))
    story.append(Paragraph(
        "• <b>Reporting:</b> We will be able to report on each model's effectiveness for stroke prediction and on how well our predictors handle the unbalanced dataset (e.g., recall, F1). We will have a clear picture of our results and assumptions.",
        bullet_style,
    ))
    story.append(Paragraph(
        "• <b>Next steps:</b> We will wrap up all experiments, begin the project report, and decide how to present our results as a coherent story.",
        bullet_style,
    ))

    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    print("Wrote proposal.pdf")

if __name__ == "__main__":
    main()
