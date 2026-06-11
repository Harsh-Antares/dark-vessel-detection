"""The killer artifact: the MPA illegal-fishing pressure map.

An interactive folium (Leaflet) HTML map showing:
  * the MPA boundary polygon(s),
  * cooperative detections (blue), dark detections (red), unknown (grey),
  * a density heat-surface of dark detections,
  * a banner with the estimated dark-vessel-hours and its 95% interval.

One clean honest map of a recognisable protected place, with ships in it
that shouldn't be there, is the artifact people remember.
"""

from __future__ import annotations

from pathlib import Path

import folium
from folium.plugins import HeatMap


def build_pressure_map(joined, mpas, estimate=None,
                       out_html: str | Path = "outputs/maps/pressure_map.html"):
    """Render the interactive map. `joined` is the detections GeoDataFrame
    (with is_dark and mpa_name), `mpas` the MPA polygons, `estimate` an
    EffortEstimate or None."""
    center = [float(joined["lat"].mean()), float(joined["lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    folium.GeoJson(
        mpas.to_json(),
        name="Marine Protected Areas",
        style_function=lambda _: {
            "color": "#2e7d32", "weight": 2, "fillColor": "#a5d6a7",
            "fillOpacity": 0.15,
        },
        tooltip=folium.GeoJsonTooltip(fields=["mpa_name"]),
    ).add_to(fmap)

    groups = {
        "dark": (folium.FeatureGroup(name="Dark vessels (no AIS)"), "#d32f2f"),
        "coop": (folium.FeatureGroup(name="Cooperative vessels (AIS)"), "#1976d2"),
        "unknown": (folium.FeatureGroup(name="Unmatched detections"), "#9e9e9e"),
    }
    heat_points = []
    for _, row in joined.iterrows():
        if row["is_dark"] == 1.0:
            key = "dark"
            heat_points.append([row["lat"], row["lon"]])
        elif row["is_dark"] == 0.0:
            key = "coop"
        else:
            key = "unknown"
        group, color = groups[key]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4, color=color, fill=True, fill_opacity=0.8,
            popup=(f"score={row['score']:.2f}<br>"
                   f"P(vessel)={row['p_vessel']:.2f}<br>"
                   f"P(fishing)={row['p_fishing']:.2f}<br>"
                   f"length≈{row['length_m']:.0f} m"),
        ).add_to(group)
    for group, _ in groups.values():
        group.add_to(fmap)

    if heat_points:
        HeatMap(heat_points, name="Dark-vessel density", radius=18,
                blur=24, min_opacity=0.3).add_to(fmap)

    if estimate is not None:
        banner = (
            f'<div style="position: fixed; top: 12px; left: 60px; z-index: 9999;'
            f' background: white; padding: 10px 14px; border: 2px solid #d32f2f;'
            f' border-radius: 6px; font-family: sans-serif; max-width: 420px;">'
            f'<b>Estimated dark-vessel-hours:</b> '
            f'{estimate.hours_mean:,.0f} '
            f'<small>[{estimate.hours_lo:,.0f}–{estimate.hours_hi:,.0f}, 95% CI]</small><br>'
            f'<small>{estimate.n_snapshots} SAR pass(es), '
            f'{estimate.window_hours:.0f} h window. Presence-hours '
            f'extrapolation — see methods note for assumptions.</small></div>'
        )
        fmap.get_root().html.add_child(folium.Element(banner))

    folium.LayerControl().add_to(fmap)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html
