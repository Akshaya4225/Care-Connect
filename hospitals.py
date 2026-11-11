import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def find_hospitals(lat, lon, radius=5000):
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"~"hospital|clinic"](around:{radius},{lat},{lon});
      way["amenity"~"hospital|clinic"](around:{radius},{lat},{lon});
      relation["amenity"~"hospital|clinic"](around:{radius},{lat},{lon});
    );
    out center tags;
    """
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=25)
    response.raise_for_status()
    elements = response.json().get("elements", [])
    results = []
    for item in elements:
        tags = item.get("tags", {})
        center = item.get("center", {})
        item_lat = item.get("lat", center.get("lat"))
        item_lon = item.get("lon", center.get("lon"))
        if item_lat is None or item_lon is None:
            continue
        results.append({
            "name": tags.get("name", "Unnamed medical facility"),
            "type": tags.get("amenity", "medical facility"),
            "lat": item_lat,
            "lon": item_lon,
            "address": ", ".join(filter(None, [
                tags.get("addr:street"), tags.get("addr:city"), tags.get("addr:postcode")
            ]))
        })
    return results[:30]
