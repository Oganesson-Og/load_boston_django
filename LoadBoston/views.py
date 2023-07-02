from django.shortcuts import render
from joblib import load
import numpy as np


model = load("./../savedModels/random_forest.joblib")


def predictor(request):
    if request.method == "POST":
        per_capita_crime_rate = request.POST["per_capita_crime_rate"]
        land_zoned_proportion = request.POST["land_zoned_proportion"]
        industry_proportion = request.POST["industry_proportion"]
        charles_river_dummy = request.POST["charles_river_dummy"]
        nitric_oxide_conc = request.POST["nitric_oxide_conc"]
        proportion_of_old = request.POST["proportion_of_old"]
        weighted_distances = request.POST["weighted_distances"]
        highway_accessibilty = request.POST["highway_accessibilty"]
        property_tax_rate = request.POST["property_tax_rate"]
        pupil_per_teacher = request.POST["pupil_per_teacher"]
        blacks_ratio = request.POST["blacks_ratio"]
        percentage_lower_status = request.POST["percentage_lower_status"]
        median_value_homes = request.POST["median_value_homes"]

        y_pred = model.predict(
            [
                [
                    per_capita_crime_rate,
                    land_zoned_proportion,
                    industry_proportion,
                    charles_river_dummy,
                    nitric_oxide_conc,
                    proportion_of_old,
                    weighted_distances,
                    highway_accessibilty,
                    property_tax_rate,
                    pupil_per_teacher,
                    blacks_ratio,
                    percentage_lower_status,
                    median_value_homes,
                ]
            ]
        )
        y_pred = y_pred[0] * 1000
        y_pred = np.round(y_pred, 2)
        y_pred = f"$ {y_pred}"
        return render(request, "main.html", {"result": y_pred})
    return render(request, "main.html")
