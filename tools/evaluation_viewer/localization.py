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
    "Use observation CSV": "Beobachtungs-CSV verwenden",
    "Open observation CSV…": "Beobachtungs-CSV öffnen…",
    "Saved point forecasts": "Gespeicherte Punktvorhersagen",
    "Forecast origin time [s]": "Zeitpunkt der Prognose [s]",
    "Runtime": "Laufzeit",
    "Comparison settings": "Vergleichseinstellungen",
    "Playback": "Wiedergabe",
    "Playback…": "Wiedergabe…",
    "Playback interval must be finite and at least 0.001 s.": "Das Wiedergabeintervall muss endlich sein und mindestens 0,001 s betragen.",
    "Delay between saved forecast origins. Smaller values play faster.": "Wartezeit zwischen gespeicherten Prognosestartpunkten. Kleinere Werte spielen schneller ab.",
    "Data sources": "Datenquellen",
    "Space": "Leertaste",
    "Toolbar": "Werkzeugleiste",
    "Start / Pause": "Start / Pause",
    "Metric curves": "Metrikverläufe",
    "N slider": "N-Schieberegler",
    "Saved forecasts": "Gespeicherte Vorhersagen",
    "H and T": "H und T",
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


HELP_SECTIONS = (
    (
        "Keyboard shortcuts",
        (
            (
                "Space",
                "Start or pause playback after loading results.",
                "Wiedergabe nach dem Laden der Ergebnisse starten oder pausieren.",
            ),
            (
                "← / →",
                "Move one saved forecast backward or forward and pause playback.",
                "Einen gespeicherten Prognosestart zurück oder vorwärts wechseln und die Wiedergabe pausieren.",
            ),
            (
                "Toolbar",
                "Pan, zoom or restore the plot view. Manual zoom is retained during playback.",
                "Ansicht verschieben, vergrößern oder zurücksetzen. Manueller Zoom bleibt bei der Wiedergabe erhalten.",
            ),
        ),
    ),
    (
        "Menu",
        (
            (
                "File",
                "Select a results directory, reload results or exit. Data selection is locked until the active analysis is cancelled.",
                "Ergebnisordner auswählen, Ergebnisse neu laden oder beenden. Die Datenwahl ist bis zum Abbruch der aktiven Analyse gesperrt.",
            ),
            (
                "View",
                "Show or hide the sidebar and open plot display settings.",
                "Seitenleiste ein- oder ausblenden und die Plot-Darstellung einstellen.",
            ),
            (
                "Settings",
                "Open Playback to set the wait between displayed forecast origins in seconds.",
                "Unter Wiedergabe die Wartezeit zwischen angezeigten Prognosestartpunkten in Sekunden einstellen.",
            ),
            (
                "Language",
                "Switch between English and German.",
                "Zwischen Englisch und Deutsch wechseln.",
            ),
            (
                "Help",
                "Open or focus this non-modal help window. Playback can continue while it is open.",
                "Dieses nichtmodale Hilfefenster öffnen oder in den Vordergrund holen. Die Wiedergabe kann dabei weiterlaufen.",
            ),
        ),
    ),
    (
        "Data sources",
        (
            (
                "Evaluation directory",
                "Read saved model evaluation CSVs with common forecast origins, horizons and reference positions. Models are not fitted again.",
                "Gespeicherte Ergebnis-CSVs mit gemeinsamen Prognosestartpunkten, Horizonten und Referenzpositionen lesen. Die Modelle werden nicht erneut geschätzt.",
            ),
            (
                "Observation CSV",
                "Optional time,x,y input for observation history and the current-position marker. Disable Use observation CSV to ignore a missing or invalid file; no IMM track is substituted.",
                "Optionale time,x,y-Datei für den Beobachtungsverlauf und den aktuellen Positionsmarker. Beobachtungs-CSV verwenden ausschalten, um eine fehlende oder ungültige Datei zu ignorieren. Ein IMM-Track wird nicht als Ersatz verwendet.",
            ),
            (
                "CSV error",
                "An enabled invalid file stops loading and preserves the previous results. Cancel analysis unlocks the source fields.",
                "Eine aktivierte ungültige Datei beendet den Ladevorgang; vorherige Ergebnisse bleiben erhalten. Analyse abbrechen gibt die Datenfelder frei.",
            ),
        ),
    ),
    (
        "Controls",
        (
            (
                "Start analysis",
                "Load the selected CSVs in the background and prepare the diagrams.",
                "Die ausgewählten CSVs im Hintergrund laden und die Diagramme vorbereiten.",
            ),
            (
                "Cancel analysis",
                "Cancel preparation or stop playback, retain loaded results and unlock the data sources. The red button remains active while playback is paused.",
                "Vorbereitung abbrechen oder Wiedergabe stoppen, geladene Ergebnisse behalten und Datenquellen freigeben. Der rote Knopf bleibt auch bei pausierter Wiedergabe aktiv.",
            ),
            (
                "Start / Pause",
                "Play or pause the saved forecast sequence. Starting again at the end restarts from the first origin.",
                "Die gespeicherte Prognosefolge abspielen oder pausieren. Ein erneuter Start am Ende beginnt beim ersten Prognosestart.",
            ),
            (
                "Playback interval [s]",
                "Minimum wait before the next displayed origin; rendering can take longer. Default 0.2 s. Apply changes the speed immediately without changing the selected origin or saved data.",
                "Mindestwartezeit bis zum nächsten angezeigten Prognosestart; das Zeichnen kann länger dauern. Standard: 0,2 s. Übernehmen ändert die Geschwindigkeit sofort, ohne Zeitposition oder gespeicherte Daten zu ändern.",
            ),
            (
                "N slider",
                "Show current/total observations through selectable forecast origins. Without observations, count positions in the common forecast sequence.",
                "Aktuelle/gesamte Beobachtungsanzahl bis zu den auswählbaren Prognosestartpunkten anzeigen. Ohne Beobachtungen werden Positionen in der gemeinsamen Prognosefolge gezählt.",
            ),
            (
                "Follow ship",
                "Follow the latest input observation while retaining zoom. Without observations, center on the nearest reference point, independently of the selected models.",
                "Der letzten Eingabebeobachtung folgen und den Zoom beibehalten. Ohne Beobachtungen am nächsten Referenzpunkt zentrieren, unabhängig von der Modellauswahl.",
            ),
        ),
    ),
    (
        "Evaluation metrics",
        (
            (
                "Metric curves",
                "Select models and metrics in the sidebar. Click a metric diagram to choose the nearest saved origin. The vertical line marks that origin; time is relative to the first common origin.",
                "Modelle und Metriken in der Seitenleiste wählen. Ein Klick ins Metrikdiagramm wählt den nächsten gespeicherten Prognosestart. Die senkrechte Linie markiert ihn; die Zeit zählt ab dem ersten gemeinsamen Prognosestart.",
            ),
            (
                "ADE / FDE / RMSE",
                "Mean position error, final position error and root mean squared position error over the future positions, in meters; smaller is better.",
                "Mittlerer Positionsfehler, Fehler am letzten Prognosepunkt und Wurzel des mittleren quadratischen Positionsfehlers über zukünftige Positionen, in Metern; kleiner ist besser.",
            ),
            (
                "Energy Score",
                "Mean saved Energy Score over the forecast horizon, in meters; smaller is better.",
                "Mittlerer gespeicherter Energy Score über den Prognosehorizont, in Metern; kleiner ist besser.",
            ),
            (
                "NLPD",
                "Mean negative log predictive density over the horizon; smaller is better. Gaussian and Student-t scores are separate choices. Student-t requires complete saved scores for every model.",
                "Mittlere negative logarithmische Vorhersagedichte über den Horizont; kleiner ist besser. Gauß- und Student-t-Scores sind getrennt wählbar. Student-t erfordert vollständige gespeicherte Scores für alle Modelle.",
            ),
            (
                "Runtime",
                "Saved fit and forecast time at each origin in seconds. Changing playback speed does not change this metric.",
                "Gespeicherte Schätz- und Vorhersagezeit je Prognosestart in Sekunden. Die Wiedergabegeschwindigkeit verändert diese Metrik nicht.",
            ),
            (
                "H and T",
                "H is the number of future positions per forecast. T in a model label is its maximum VI/MCMC observation-window size.",
                "H ist die Anzahl zukünftiger Positionen je Prognose. T in einer Modellbeschriftung ist die maximale Beobachtungsfenstergröße von VI/MCMC.",
            ),
        ),
    ),
    (
        "Plot display",
        (
            (
                "Plot layout",
                "Default: one full-width metric above the trajectory. Disable One metric above trajectory for a trajectory on the left and two metrics on the right.",
                "Standard: ein breites Metrikdiagramm über der Trajektorie. Ein Metrikdiagramm über der Trajektorie ausschalten, um links die Trajektorie und rechts zwei Metriken zu sehen.",
            ),
            (
                "Metric time axis",
                "Show the complete history or enable Follow metric time with a moving window. Window width uses seconds.",
                "Den gesamten Verlauf anzeigen oder mit Metrikzeit verfolgen ein mitlaufendes Fenster verwenden. Die Fensterbreite wird in Sekunden angegeben.",
            ),
            (
                "Trajectory",
                "Toggle the reference, its forecast segment, reference points, observations and the current-position marker. Trajectory and metric legends can be switched separately.",
                "Referenz, Referenz im Prognosehorizont, Referenzpunkte, Beobachtungen und aktuellen Positionsmarker ein- oder ausblenden. Trajektorien- und Metriklegende sind getrennt schaltbar.",
            ),
            (
                "Saved forecasts",
                "Model lines contain only saved future points and keep each run's mean or median. They are not connected to the current marker. Errors belong to forecast origins and may already reflect a coming bend.",
                "Modelllinien enthalten nur gespeicherte zukünftige Punkte und behalten Mittelwert oder Median des jeweiligen Laufs bei. Sie werden nicht mit dem aktuellen Marker verbunden. Fehler gehören zum Prognosestart und können bereits eine bevorstehende Kurve widerspiegeln.",
            ),
        ),
    ),
)


def help_sections(language="en"):
    """Return localized headings and rows for the scrollable help window."""
    return tuple(
        (
            text(heading, language),
            tuple(
                (text(label, language), german if language == "de" else english)
                for label, english, german in rows
            ),
        )
        for heading, rows in HELP_SECTIONS
    )


def text(source, language="en"):
    if source == "Help text" and language == "en":
        return ENGLISH_HELP
    return (
        GERMAN.get(source, translate(source, language)) if language == "de" else source
    )
