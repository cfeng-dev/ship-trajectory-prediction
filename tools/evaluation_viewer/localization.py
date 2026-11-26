"""Reuse Predictor translations and add the viewer's own presentation text."""

from bayesian_predictor.localization import translate

GERMAN = {
    "Evaluation Viewer": "Evaluation Viewer",
    "Model evaluation": "Modellevaluation",
    "Evaluation directory": "Ergebnisordner",
    "Open directory…": "Ordner öffnen…",
    "Reload": "Neu laden",
    "Load CSVs": "CSVs laden",
    "Models": "Modelle",
    "Metric diagrams": "Metrikdiagramme",
    "Upper diagram": "Oberes Diagramm",
    "Lower diagram": "Unteres Diagramm",
    "Metric time axis": "Metrik-Zeitachse",
    "Follow metric time": "Metrikzeit verfolgen",
    "Window width [s]": "Fensterbreite [s]",
    "Trajectory view": "Trajektorienansicht",
    "View span [m]": "Ansichtsbreite [m]",
    "Forecast origin": "Prognosestart",
    "Matched origins": "Gemeinsame Prognosestartpunkte",
    "Select a directory with saved evaluation CSVs.": "Wähle einen Ordner mit gespeicherten Ergebnis-CSVs.",
    "No data loaded": "Keine Daten geladen",
    "Select at least one model.": "Wähle mindestens ein Modell.",
    "Select an available metric for each diagram.": "Wähle für jedes Diagramm eine verfügbare Metrik.",
    "Reference forecast segment": "Referenz im Prognosehorizont",
    "{model} track at forecast origin": "{model}-Track am Prognosestart",
    "Connection to IMM track": "Verbindung zum IMM-Track",
    "Connection to current position": "Verbindung zur aktuellen Position",
    "{model} track through forecast origin": "{model}-Track bis zum Prognosestart",
    "Latest observation": "Letzte Beobachtung",
    "Observation CSV": "Beobachtungs-CSV",
    "Open observation CSV…": "Beobachtungs-CSV öffnen…",
    "Saved point forecasts": "Gespeicherte Punktvorhersagen",
    "Forecast origin time [s]": "Zeitpunkt der Prognose [s]",
    "Runtime": "Laufzeit",
    "Comparison settings": "Vergleichseinstellungen",
    "CSV error": "CSV-Fehler",
    "Invalid settings": "Ungültige Einstellungen",
    "Follow ship": "Schiff verfolgen",
    "Reference trajectory": "Referenztrajektorie",
    "Easting x [m]": "Ostkoordinate x [m]",
    "Northing y [m]": "Nordkoordinate y [m]",
    "Help text": "Lade einen Ergebnisordner, wähle Modelle und je Diagramm eine Metrik und klicke auf Anwenden.\n\nZeitschieber, Pfeiltasten und Start/Pause steuern denselben Prognosestartpunkt. Ein Klick in ein Metrikdiagramm wählt den nächstgelegenen gespeicherten Zeitpunkt.\n\nDie Linien zeigen vollständige gespeicherte Verläufe im sichtbaren Zeitfenster. Ihre Farben entsprechen den Modellen. Die senkrechte Linie markiert den ausgewählten Prognosestartpunkt.\n\nDie linke Ansicht zeigt gespeicherte Punktvorhersagen; sie rekonstruiert keine Sample-Trajektorien. Fehler sind dem Prognosestart zugeordnet: Die vorhergesagten Positionen können bereits in einer bevorstehenden Kurve liegen.",
}
ENGLISH_HELP = (
    "Load a results directory, select models and one metric per diagram, then choose Apply.\n\n"
    "Use the time slider, arrow keys and Start/Pause to select a common forecast origin. "
    "Click a metric diagram to select the nearest saved origin.\n\n"
    "Lines show the complete saved histories in the visible time window. Colors identify models. "
    "The vertical line marks the selected forecast origin.\n\n"
    "The trajectory view shows saved point forecasts; it does not reconstruct sample trajectories. "
    "Errors are indexed by forecast origin: a forecast may already extend into an upcoming bend."
)


def text(source, language="en"):
    if source == "Help text" and language == "en":
        return ENGLISH_HELP
    return (
        GERMAN.get(source, translate(source, language)) if language == "de" else source
    )
