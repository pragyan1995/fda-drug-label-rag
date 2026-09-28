import requests
import os

os.makedirs("dailymed_data", exist_ok=True)

drug_names = ["aspirin", "ibuprofen", "metformin"]

for drug in drug_names:
    search_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={drug}&pagesize=3"
    resp = requests.get(search_url)
    data = resp.json()

    for entry in data.get("data", []):
        setid = entry["setid"]
        xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
        xml_resp = requests.get(xml_url)

        filename = f"dailymed_data/{drug}_{setid}.xml"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(xml_resp.text)
        print(f"Saved: {filename}")