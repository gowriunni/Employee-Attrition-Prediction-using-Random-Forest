

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved preprocessing + Random Forest pipeline
model = joblib.load("attrition_model.joblib")

# Valid choices for categorical features
choices = {
    "BusinessTravel": [
        "Non-Travel",
        "Travel_Rarely",
        "Travel_Frequently"
    ],
    "Department": [
        "Sales",
        "Research & Development",
        "Human Resources"
    ],
    "EducationField": [
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Other",
        "Human Resources"
    ],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"]
}

# Numerical feature defaults and reasonable input ranges
numeric_fields = {
    "Age": (30, 18, 60),
    "DailyRate": (800, 100, 1500),
    "DistanceFromHome": (5, 1, 29),
    "Education": (3, 1, 5),
    "EnvironmentSatisfaction": (3, 1, 4),
    "HourlyRate": (65, 30, 100),
    "JobInvolvement": (3, 1, 4),
    "JobLevel": (2, 1, 5),
    "JobSatisfaction": (3, 1, 4),
    "MonthlyIncome": (4000, 1000, 20000),
    "MonthlyRate": (14000, 2000, 27000),
    "NumCompaniesWorked": (2, 0, 9),
    "PercentSalaryHike": (15, 11, 25),
    "PerformanceRating": (3, 3, 4),
    "RelationshipSatisfaction": (3, 1, 4),
    "StockOptionLevel": (1, 0, 3),
    "TotalWorkingYears": (8, 0, 40),
    "TrainingTimesLastYear": (3, 0, 6),
    "WorkLifeBalance": (3, 1, 4),
    "YearsAtCompany": (5, 0, 40),
    "YearsInCurrentRole": (3, 0, 18),
    "YearsSinceLastPromotion": (1, 0, 15),
    "YearsWithCurrManager": (3, 0, 17)
}

# Build the form using the exact columns expected by the model
features = []

for column in model.feature_names_in_:
    if column in choices:
        features.append({
            "name": column,
            "kind": "choice",
            "options": choices[column]
        })
    elif column in numeric_fields:
        default, minimum, maximum = numeric_fields[column]
        features.append({
            "name": column,
            "kind": "number",
            "default": default,
            "minimum": minimum,
            "maximum": maximum
        })
    else:
        raise ValueError(
            f"Please configure the input field: {column}"
        )


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    # Set defaults for the form
    values = {
        field["name"]: (
            field["options"][0]
            if field["kind"] == "choice"
            else field["default"]
        )
        for field in features
    }

    if request.method == "POST":
        try:
            row = {}

            for field in features:
                name = field["name"]
                raw_value = request.form.get(name, "").strip()

                if not raw_value:
                    raise ValueError(f"{name} is required.")

                if field["kind"] == "choice":
                    if raw_value not in field["options"]:
                        raise ValueError(
                            f"Invalid selection for {name}."
                        )
                    row[name] = raw_value
                    values[name] = raw_value

                else:
                    number = int(raw_value)

                    if not (
                        field["minimum"]
                        <= number
                        <= field["maximum"]
                    ):
                        raise ValueError(
                            f"{name} must be between "
                            f"{field['minimum']} and "
                            f"{field['maximum']}."
                        )

                    row[name] = number
                    values[name] = number

            # Preserve the model's expected column order
            input_df = pd.DataFrame(
                [row],
                columns=model.feature_names_in_
            )

            # Run prediction through the saved pipeline
            result = model.predict(input_df)[0]

            class_labels = list(model.classes_)
            probabilities = model.predict_proba(input_df)[0]
            yes_index = class_labels.index("Yes")

            probability = round(
                float(probabilities[yes_index]) * 100, 2
            )

            prediction = (
                "Higher predicted attrition risk"
                if result == "Yes"
                else "Lower predicted attrition risk"
            )

        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        features=features,
        values=values,
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
