"""Reuse Predictor translations and add the viewer's own presentation text."""

from bayesian_predictor.localization import translate

GERMAN = {
    "Evaluation Viewer": "Evaluation Viewer",
    "Model evaluation": "Modellevaluation",
    "Evaluation directory": "Ergebnisordner",
    "Open directory…": "Ordner öffnen…",
    "Reload": "Neu laden",
    "Choose the input paths, then click Start analysis.": "Wähle die Eingabepfade und klicke auf Analyse starten.",
    "Loading evaluation CSVs …": "Ergebnis-CSVs werden geladen …",
    "Loading observations …": "Beobachtungen werden geladen …",
    "Preparing metrics …": "Metriken werden vorbereitet …",
    "Preparing diagrams …": "Diagramme werden vorbereitet …",
    "Models": "Modelle",
    "Metric diagrams": "Metrikdiagramme",
    "General": "Allgemein",
    "Show trajectory legend": "Trajektorienlegende anzeigen",
    "Show metric legend": "Metriklegende anzeigen",
    "Trajectory": "Trajektorie",
    "Plot layout": "Plot-Aufteilung",
    "One metric above trajectory": "Ein Metrikdiagramm über der Trajektorie",
    "Metric": "Metrik",
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
    "Show reference forecast segment": "Referenz im Prognosehorizont anzeigen",
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
    "Help text": "Wähle den Ergebnisordner und die optionale Beobachtungs-CSV und klicke auf Analyse starten. Während des Ladens erscheint der Fortschritt mittig; Analyse abbrechen verwirft den laufenden Ladevorgang. Änderungen an Modellen und Metriken aktualisieren die Diagramme sofort. Mindestens ein Modell muss ausgewählt bleiben. Ein Metrikwechsel erhält die Zeitposition und lässt eine laufende Wiedergabe weiterlaufen.\n\nZeitschieber, Pfeiltasten und Start/Pause steuern denselben Prognosestartpunkt. Ein Klick in ein Metrikdiagramm wählt den nächstgelegenen gespeicherten Zeitpunkt.\n\nDie Linien zeigen vollständige gespeicherte Verläufe im sichtbaren Zeitfenster. Ihre Farben entsprechen den Modellen. Die senkrechte Linie markiert den ausgewählten Prognosestartpunkt.\n\nDie linke Ansicht zeigt gespeicherte Punktvorhersagen; sie rekonstruiert keine Sample-Trajektorien. Fehler sind dem Prognosestart zugeordnet: Die vorhergesagten Positionen können bereits in einer bevorstehenden Kurve liegen.",
}
ENGLISH_HELP = (
    "Choose a results directory and optional observation CSV, then click Start analysis. "
    "Progress appears in the center; Cancel analysis discards the pending load. "
    "Model and metric selections update the plots immediately; at least one model must remain selected. "
    "Metric changes preserve the current origin and continue active playback.\n\n"
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
