"""Presentation-only English/German translations owned by Trajectory Predictor."""

import re
from string import Formatter

from matplotlib.text import Text

TRANSLATIONS = {
    "Bayesian CTRV — Ship Trajectory Predictor": "Bayessches CTRV — Schiffstrajektorienprognose",
    "Trajectory Predictor": "Trajektorienprognose",
    "File": "Datei",
    "Open CSV…": "CSV öffnen…",
    "Exit": "Beenden",
    "View": "Ansicht",
    "Settings": "Einstellungen",
    "Help": "Hilfe",
    "Show Help": "Hilfe anzeigen",
    "Language": "Sprache",
    "English": "English",
    "Deutsch": "Deutsch",
    "Local [m]": "Lokal [m]",
    "Local [km]": "Lokal [km]",
    "Priors…": "Prior-Verteilungen…",
    "Inference parameters…": "Inferenzparameter…",
    "Analysis setup…": "Analyseeinstellungen…",
    "Plot display…": "Plot-Darstellung…",
    "Posterior analysis": "Posterior-Analyse",
    "Analysis": "Analyse",
    "Start analysis": "Analyse starten",
    "Cancel analysis": "Analyse abbrechen",
    "Data": "Daten",
    "CSV file": "CSV-Datei",
    "Select CSV file": "CSV-Datei auswählen",
    "Select trajectory CSV": "Trajektorien-CSV auswählen",
    "CSV files": "CSV-Dateien",
    "All files": "Alle Dateien",
    "Inference method": "Inferenzverfahren",
    "SMC – Sequential Monte Carlo": "SMC – Sequenzielles Monte Carlo",
    "Seq. VI – Sequential variational inference": "Seq. VI – Sequenzielle Variationsinferenz",
    "VI – Variational inference": "VI – Variationsinferenz",
    "MCMC – Markov chain Monte Carlo": "MCMC – Markov-Chain-Monte-Carlo",
    "Reference state": "Referenzzustand",
    "Posterior medians": "Posterior-Mediane",
    "Position": "Position",
    "Speed": "Geschwindigkeit",
    "Heading": "Kurswinkel",
    "Turn rate": "Drehrate",
    "Time": "Zeit",
    "Position-observation noise": "Beobachtungsrauschen",
    "Speed-process noise": "Geschwindigkeitsrauschen",
    "Turn-rate-process noise": "Drehratenrauschen",
    "Evaluation metrics": "Evaluationsmetriken",
    "Forecast ADE (median)": "Prognose-ADE (Median)",
    "Forecast FDE (median)": "Prognose-FDE (Median)",
    "Forecast RMSE (median)": "Prognose-RMSE (Median)",
    "Forecast Energy Score": "Prognose-Energy-Score",
    "Joint 50% coverage": "2D-Abdeckung 50%",
    "Joint 90% coverage": "2D-Abdeckung 90%",
    "Inference time": "Inferenzzeit",
    "BR24 radar association": "BR24-Radarzuordnung",
    "Cycle": "Zyklus",
    "Detection ID": "Detektions-ID",
    "Initial measurement": "Erste Messung",
    "Associated": "Zugeordnet",
    "Prediction only": "Nur Prädiktion",
    "Priors": "Prior-Verteilungen",
    "Analysis setup": "Analyseeinstellungen",
    "Inference parameters — {method}": "Inferenzparameter — {method}",
    "Plot display": "Plot-Darstellung",
    "Apply": "Übernehmen",
    "Cancel": "Abbrechen",
    "Check settings": "Einstellungen prüfen",
    "Coordinate display": "Koordinatendarstellung",
    "Start index (from 0)": "Startindex (ab 0)",
    "Observation interval [s] (empty = original)": "Beobachtungsintervall [s] (leer = original)",
    "Maximum observations (empty = all)": "Maximale Beobachtungen (leer = alle)",
    "Additional position noise [m]": "Zusätzliches Positionsrauschen [m]",
    "Position-noise seed": "Zufallsstartwert für Positionsrauschen",
    "Forecast steps (0 = off)": "Prognoseschritte (0 = aus)",
    "Future trajectories (0 = off)": "Zukünftige Trajektorien (0 = aus)",
    "Inference seed": "Zufallsstartwert für Inferenz",
    "Playback interval [s]": "Wiedergabeintervall [s]",
    "Show legend": "Legende anzeigen",
    "Show reference trajectory": "Referenztrajektorie anzeigen",
    "Show observations / Bayesian track through N": "Beobachtungen / Bayesschen Track bis N anzeigen",
    "Show current position": "Aktuelle Position anzeigen",
    "Show possible future trajectories": "Mögliche zukünftige Trajektorien anzeigen",
    "Connect samples to current position": "Samples mit aktueller Position verbinden",
    "Show forecast (median)": "Prognose (Median) anzeigen",
    "Show forecast (mean)": "Prognose (Mittelwert) anzeigen",
    "Show 50% posterior-predictive region": "50%-Posterior-Prädiktionsregion anzeigen",
    "Show 90% posterior-predictive region": "90%-Posterior-Prädiktionsregion anzeigen",
    "Prediction uncertainty": "Prognoseunsicherheit",
    "Follow-ship view span [m]": "Ansichtsgröße beim Verfolgen [m]",
    "Speed: upper threshold [m/s]": "Geschwindigkeit: oberer Schwellenwert [m/s]",
    "Speed: tail probability": "Geschwindigkeit: Überschreitungswahrscheinlichkeit",
    "Turn rate: threshold [°/s]": "Drehrate: Schwellenwert [°/s]",
    "Turn rate: tail probability": "Drehrate: Überschreitungswahrscheinlichkeit",
    "Additional noise: threshold [m]": "Zusatzrauschen: Schwellenwert [m]",
    "Additional noise: tail probability": "Zusatzrauschen: Überschreitungswahrscheinlichkeit",
    "Effective observation-noise floor [m]": "Minimales effektives Beobachtungsrauschen [m]",
    "Speed process: threshold [m/s]": "Geschwindigkeitsprozess: Schwellenwert [m/s]",
    "Speed process: tail probability": "Geschwindigkeitsprozess: Überschreitungswahrscheinlichkeit",
    "Turn-rate process: threshold [°/s]": "Drehratenprozess: Schwellenwert [°/s]",
    "Turn-rate process: tail probability": "Drehratenprozess: Überschreitungswahrscheinlichkeit",
    "Particle count": "Partikelanzahl",
    "Posterior draws": "Posterior-Samples",
    "Resampling: ESS fraction": "Resampling: ESS-Anteil",
    "Rejuvenation scale": "Skalierung der Partikelverjüngung",
    "Require VI convergence": "VI-Konvergenz voraussetzen",
    "Bootstrap observations": "Bootstrap-Beobachtungen",
    "VI algorithm": "VI-Algorithmus",
    "Maximum VI iterations": "Maximale VI-Iterationen",
    "Gradient samples": "Gradienten-Samples",
    "ELBO samples": "ELBO-Samples",
    "VI learning rate": "VI-Lernrate",
    "VI adaptation iterations": "VI-Adaptionsiterationen",
    "Relative ELBO tolerance": "Relative ELBO-Toleranz",
    "ELBO evaluation interval": "ELBO-Auswertungsintervall",
    "Show CmdStan console": "CmdStan-Konsole anzeigen",
    "Covariance jitter": "Kovarianz-Jitter",
    "Covariance regularization attempts": "Versuche der Kovarianzregularisierung",
    "Speed — half-normal distribution": "Geschwindigkeit — Halbnormalverteilung",
    "Heading — uniform distribution": "Kurswinkel — Gleichverteilung",
    "Heading — uniform distribution (−180° to 180°)": "Kurswinkel — Gleichverteilung (−180° bis 180°)",
    "Turn rate — normal distribution": "Drehrate — Normalverteilung",
    "Turn rate — normal distribution around 0": "Drehrate — Normalverteilung um 0",
    "Additional observation noise — exponential distribution": "Zusätzliches Beobachtungsrauschen — Exponentialverteilung",
    "Speed process noise — exponential distribution": "Geschwindigkeitsprozessrauschen — Exponentialverteilung",
    "Turn-rate process noise — exponential distribution": "Drehratenprozessrauschen — Exponentialverteilung",
    "Apply updates these settings.\nA running analysis remains unchanged until “Start analysis”.": "Übernehmen speichert diese Einstellungen.\nSie gelten nach dem nächsten Klick auf „Analyse starten“.",
    "Changes are applied only after selecting Apply.": "Änderungen werden erst mit „Übernehmen“ angewendet.",
    "Trajectory and posterior evolution\n\nSelect CSV, run, and method on the left.\nThen click “Start analysis”.": "Trajektorie und Posterior-Entwicklung\n\nLinks CSV-Datei und Inferenzverfahren auswählen.\nAnschließend auf „Analyse starten“ klicken.",
    "Ready. No analysis started yet.": "Bereit. Noch keine Analyse gestartet.",
    "Preparing analysis …": "Analyse wird vorbereitet …",
    "Loading data / initializing inference. Any previous fit is stopped first.": "Daten werden geladen / Inferenz wird initialisiert. Eine vorherige Berechnung wird zuerst beendet.",
    "Analysis cancelled. Settings unlocked.": "Analyse abgebrochen. Einstellungen sind wieder verfügbar.",
    "Analysis failed": "Analyse fehlgeschlagen",
    "Analysis failed. Check settings.": "Analyse fehlgeschlagen. Bitte Einstellungen prüfen.",
    "Playing": "Wiedergabe läuft",
    "Paused": "Pausiert",
    "Computing": "Berechnung",
    "Target": "Ziel",
    "Error: {detail} — Start a new analysis to continue.": "Fehler: {detail} — Zum Fortfahren eine neue Analyse starten.",
    "Closing … a running calculation is finishing.": "Wird geschlossen … eine laufende Berechnung wird beendet.",
    "Follow ship": "Schiff verfolgen",
    "Show settings": "Einstellungen anzeigen",
    "Motion state": "Bewegungszustand",
    "Uncertainties": "Unsicherheiten",
    "Posterior group": "Posterior-Gruppe",
    "Restart": "Neu starten",
    "Reference trajectory": "Referenztrajektorie",
    "Observations through N": "Beobachtungen bis N",
    "Bayesian track through N": "Bayesscher Track bis N",
    "Current position": "Aktuelle Position",
    "Possible future trajectories": "Mögliche zukünftige Trajektorien",
    "Possible future trajectories (n = {count})": "Mögliche zukünftige Trajektorien (n = {count})",
    "Forecast (median)": "Prognose (Median)",
    "Forecast (mean)": "Prognose (Mittelwert)",
    "Horizon density region": "Dichteregion über den Prognosehorizont",
    "Horizon density contours": "Dichtekonturen über den Prognosehorizont",
    "Per-step density regions": "Dichteregionen je Zeitschritt",
    "Per-step density contours": "Dichtekonturen je Zeitschritt",
    "Covariance ellipses": "Kovarianzellipsen",
    "No uncertainty region": "Keine Unsicherheitsregion",
    "Horizon density region ({percent} %)": "Dichteregion über den Prognosehorizont ({percent} %)",
    "Horizon density contour ({percent} %)": "Dichtekontur über den Prognosehorizont ({percent} %)",
    "Per-step density region ({percent} %)": "Dichteregion je Zeitschritt ({percent} %)",
    "Per-step density contour ({percent} %)": "Dichtekontur je Zeitschritt ({percent} %)",
    "Covariance ellipse ({percent} %)": "Kovarianzellipse ({percent} %)",
    "Bayesian CTRV posterior update — N = {count}": "Bayessches CTRV: Posterior-Update — N = {count}",
    "Bayesian CTRV prior predictive — N = {count}": "Bayessches CTRV: Prior-Prädiktion — N = {count}",
    "Bayesian CTRV prior prediction — N = {count}": "Bayessches CTRV: Prior-Prognose — N = {count}",
    "Ship movement": "Schiffsbewegung",
    "Ship movement · Forecast +{seconds} s": "Schiffsbewegung · Prognose +{seconds} s",
    "Forecast from N = {count}": "Prognose ab N = {count}",
    "End of recorded trajectory: no further forecast times.": "Ende der Referenztrajektorie: keine weiteren Prognosezeitpunkte.",
    "Mean forecast unavailable: no posterior draws.": "Mittelwertprognose nicht verfügbar: keine Posterior-Samples.",
    "Speed posterior": "Posterior der Geschwindigkeit",
    "Heading posterior": "Posterior des Kurswinkels",
    "Turn-rate posterior": "Posterior der Drehrate",
    "Position-observation-noise posterior": "Posterior des Beobachtungsrauschens",
    "Speed-process-noise posterior": "Posterior des Geschwindigkeitsrauschens",
    "Turn-rate-process-noise posterior": "Posterior des Drehratenrauschens",
    "Initial speed prior": "Anfangsprior der Geschwindigkeit",
    "Initial heading prior": "Anfangsprior des Kurswinkels",
    "Initial turn-rate prior": "Anfangsprior der Drehrate",
    "Initial position-observation-noise prior": "Anfangsprior des Beobachtungsrauschens",
    "Initial speed-process-noise prior": "Anfangsprior des Geschwindigkeitsrauschens",
    "Initial turn-rate-process-noise prior": "Anfangsprior des Drehratenrauschens",
    "Initial prior": "Anfangsprior",
    "Posterior density": "Posterior-Dichte",
    "Density": "Dichte",
    "Easting x [m]": "Ostkoordinate x [m]",
    "Northing y [m]": "Nordkoordinate y [m]",
    "Easting x [km]": "Ostkoordinate x [km]",
    "Northing y [km]": "Nordkoordinate y [km]",
    "chains": "MCMC-Ketten",
    "parallel_chains": "Parallele MCMC-Ketten",
    "iter_sampling": "Sampling-Iterationen",
    "iter_warmup": "Warmup-Iterationen",
    "max_treedepth": "Maximale Baumtiefe",
    "Invalid value for {field}.": "Ungültiger Wert für {field}.",
    "{field} must be at least {minimum}.": "{field} muss mindestens {minimum} betragen.",
    "{field} must be greater than 0.": "{field} muss größer als 0 sein.",
    "VI algorithm must be meanfield or fullrank.": "Der VI-Algorithmus muss meanfield oder fullrank sein.",
    "At least two MCMC draws are required.": "Mindestens zwei MCMC-Samples sind erforderlich.",
    "adapt_delta must be between 0 and 1.": "adapt_delta muss zwischen 0 und 1 liegen.",
    "parallel_chains must not exceed chains.": "parallel_chains darf chains nicht überschreiten.",
    "Bootstrap observations must be at least 3.": "Mindestens drei Bootstrap-Beobachtungen sind erforderlich.",
    "Sequential VI algorithm must be fullrank.": "Sequenzielle VI erfordert den Algorithmus fullrank.",
    "Posterior draws must be at least 2.": "Mindestens zwei Posterior-Samples sind erforderlich.",
    "Maximum observations must be an integer.": "Die maximale Beobachtungsanzahl muss ganzzahlig sein.",
    "Select an existing CSV file.": "Eine vorhandene CSV-Datei auswählen.",
    "Inference method must be smc, sequential_vi, vi, or mcmc.": "Das Inferenzverfahren muss smc, sequential_vi, vi oder mcmc sein.",
    "Unknown settings group: {group}": "Unbekannter Einstellungsbereich: {group}",
}

TRANSLATIONS.update(
    {
        "Keyboard shortcuts": "Tastenkürzel",
        "Space": "Leertaste",
        "Toolbar": "Werkzeugleiste",
        "Menu": "Menü",
        "Controls": "Bedienelemente",
        "Input and state displays": "Eingabedaten und Zustandsanzeigen",
        "Priors and inference": "Prior-Verteilungen und Inferenz",
        "Display": "Darstellung",
        "File → Open CSV": "Datei → CSV öffnen",
        "File → Exit": "Datei → Beenden",
        "Main-window X": "Schließen des Hauptfensters",
        "N slider": "N-Schieberegler",
        "Early updates": "Frühe Updates",
        "Analysis errors": "Analysefehler",
        "Diagnostics": "Diagnose",
        "Start index": "Startindex",
        "Observation interval": "Beobachtungsintervall",
        "Maximum observations": "Maximale Beobachtungen",
        "Additional position noise": "Zusätzliches Positionsrauschen",
        "Forecast steps": "Prognoseschritte",
        "Future trajectories": "Zukünftige Trajektorien",
        "Playback interval": "Wiedergabeintervall",
        "CSV input": "CSV-Eingabe",
        "Reference motion": "Referenzbewegung",
        "Optional CSV columns": "Optionale CSV-Spalten",
        "Reference turn rate": "Referenzdrehrate",
        "Reference boundary": "Verwendung der Referenzdaten",
        "Missing values": "Fehlende Werte",
        "Units and time": "Einheiten und Zeit",
        "Observation-noise floor": "Untergrenze des Beobachtungsrauschens",
        "Sequential VI": "Sequenzielle VI",
        "Inference parameters": "Inferenzparameter",
        "Apply / Cancel": "Übernehmen / Abbrechen",
        "Radar input": "Radareingabe",
        "No reference labels": "Fehlende Referenzdaten",
        "Blue trajectory": "Blaue Trajektorie",
        "Forecast metrics": "Prognosemetriken",
        "Connect samples": "Samples verbinden",
        "Follow-ship view span": "Ansichtsgröße beim Verfolgen",
        "50% / 90% regions": "50%- / 90%-Regionen",
        "Coverage versus display": "Abdeckung und Darstellung",
        "Start or pause plot playback after starting analysis": "Nach dem Analysestart die Plot-Wiedergabe starten oder pausieren",
        "Move one observation backward or forward and pause playback": "Eine Beobachtung zurück- oder vorgehen und die Wiedergabe pausieren",
        "Pan, zoom, or reset the plot view": "Die Plot-Ansicht verschieben, vergrößern oder zurücksetzen",
        "Select the CSV file for analysis": "CSV-Datei für die Analyse auswählen",
        "Exit Trajectory Predictor": "Trajektorienprognose beenden",
        "Show or hide settings; select meters or kilometers; open plot display options": "Einstellungen ein- oder ausblenden, Meter oder Kilometer wählen und die Plot-Darstellung öffnen",
        "Configure priors, inference, and analysis setup; locked during analysis": "Prior-Verteilungen, Inferenz und Analyse konfigurieren; während der Analyse gesperrt",
        "Configure priors, inference and analysis setup": "Prior-Verteilungen, Inferenz und Analyse konfigurieren",
        "Switch between English and German without restarting the analysis": "Zwischen Englisch und Deutsch wechseln, ohne die Analyse neu zu starten",
        "Open or focus this non-modal help window": "Dieses nicht modale Hilfefenster öffnen oder in den Vordergrund holen",
        "Close child windows and stop analysis; a running calculation finishes before exit": "Unterfenster schließen und die Analyse beenden; eine laufende Berechnung wird vor dem Beenden abgeschlossen",
        "Analyse the selected CSV with the current settings": "Ausgewählte CSV-Datei mit den aktuellen Einstellungen analysieren",
        "Discard displayed results and unlock settings; an ongoing fit may finish in the background": "Angezeigte Ergebnisse verwerfen und Einstellungen freigeben; eine laufende Anpassung kann im Hintergrund fertig werden",
        "Select SMC, Sequential VI, VI, or MCMC for the next analysis": "SMC, sequenzielle VI, VI oder MCMC für die nächste Analyse auswählen",
        "Refit the latest {count} steps at each update; BR24 windows count radar cycles, including misses": "Bei jedem Update die letzten {count} Schritte neu anpassen; bei BR24 zählen Radarzyklen einschließlich fehlender Zuordnungen",
        "Center the plot on the current position while retaining manual zoom": "Die Ansicht auf die aktuelle Position zentrieren; manuelles Zoomen bleibt möglich",
        "Show or hide the left sidebar without restarting analysis": "Linke Seitenleiste ein- oder ausblenden, ohne die Analyse neu zu starten",
        "Show the prior at N = 0 and posterior updates afterward": "Bei N = 0 die Prior-Verteilung und danach die Posterior-Updates anzeigen",
        "With too few observations, forecasts may remain prior-predictive; the plot title identifies this phase": "Bei zu wenigen Beobachtungen können Prognosen noch aus der Prior-Prädiktion stammen; der Plot-Titel kennzeichnet diese Phase",
        "Playback stops and settings unlock after an error. Start a new analysis before continuing; failed filter states are not reused": "Nach einem Fehler stoppt die Wiedergabe und die Einstellungen werden freigegeben. Zum Fortfahren eine neue Analyse starten; fehlerhafte Filterzustände werden nicht wiederverwendet",
        "Error details are printed to the launching terminal. No diagnostic log files are created": "Fehlerdetails erscheinen im startenden Terminal. Es werden keine Diagnose-Logdateien erzeugt",
        "First CSV position to include; counting begins at 0": "Erste einzubeziehende CSV-Position; die Zählung beginnt bei 0",
        "Optional minimum position spacing in seconds; leave empty to use every original timestamp": "Optionaler Mindestabstand zwischen Positionen in Sekunden; leer lassen, um alle ursprünglichen Zeitstempel zu verwenden",
        "Limit selected positions; leave empty for all": "Anzahl ausgewählter Positionen begrenzen; leer lassen für alle",
        "Gaussian standard deviation per local position axis in meters; 0 keeps processed-trajectory positions unchanged": "Gaußsche Standardabweichung je lokaler Positionsachse in Metern; bei 0 bleiben die aufbereiteten Positionen unverändert",
        "Seed used for the additional position noise": "Zufallsstartwert für das zusätzliche Positionsrauschen",
        "Number of steps predicted after each position; 0 disables forecasts": "Anzahl vorhergesagter Schritte nach jeder Position; 0 deaktiviert Prognosen",
        "Number of posterior forecast samples to draw; 0 hides them": "Anzahl gezeichneter Posterior-Prognosepfade; 0 blendet sie aus",
        "Seed used to make stochastic inference reproducible": "Zufallsstartwert für reproduzierbare stochastische Inferenz",
        "Minimum wait before requesting the next N; computation can take longer": "Mindestwartezeit bis zur Anforderung des nächsten N; die Berechnung kann länger dauern",
        "Load a processed trajectory with time, x, y in seconds and local meters, or a BR24 radar tracking CSV": "Aufbereitete Trajektorie mit time, x, y in Sekunden und lokalen Metern oder eine BR24-Radar-CSV laden",
        "Unmodified reference position and time; motion values come from reference fields or position differences, not the posterior": "Unveränderte Referenzposition und Zeit; Bewegungsgrößen stammen aus Referenzfeldern oder Positionsdifferenzen, nicht aus dem Posterior",
        "Supplied speed, yaw and turn rate take priority independently, including zero values. Missing quantities are marked (derived). Speed/course need two positions; course-change rate needs three. Course is undefined at rest and is not vessel orientation": "Vorhandene Geschwindigkeit, Orientierung und Drehrate werden jeweils bevorzugt, auch bei Nullwerten. Fehlende Größen werden als (abgeleitet) markiert. Geschwindigkeit und Kurs benötigen zwei Positionen, die Kursänderungsrate drei. Im Stillstand ist der Kurs undefiniert und entspricht nicht der Schiffsausrichtung",
        "BR24: ground_truth_velocity [m/s], ground_truth_yaw [rad], ground_truth_turn_rate [rad/s]. Processed time,x,y files also accept speed/velocity, theta/yaw and omega/turn_rate. Invalid supplied values are reported, not replaced": "BR24: ground_truth_velocity [m/s], ground_truth_yaw [rad], ground_truth_turn_rate [rad/s]. Aufbereitete time,x,y-Dateien erlauben auch speed/velocity, theta/yaw und omega/turn_rate. Ungültige vorhandene Werte werden gemeldet und nicht ersetzt",
        "Use supplied reference rates when available; otherwise derive backward wrapped yaw differences, or course differences if yaw is absent, using actual time spacing. Derived values are marked (derived) and displayed in degrees/s; unavailable values show a dash": "Vorhandene Referenzdrehraten verwenden; sonst aus rückwärts gerichteten, winkelkorrigierten Orientierungsdifferenzen oder bei fehlender Orientierung aus Kursdifferenzen mit tatsächlichen Zeitabständen ableiten. Abgeleitete Werte sind als (abgeleitet) markiert und werden in Grad/s angezeigt; fehlende Werte als Strich",
        "Estimated speed, heading, turn rate and noise levels after conditioning on the available observations": "Geschätzte Geschwindigkeit, Kurswinkel, Drehrate und Rauschgrößen unter Berücksichtigung der verfügbaren Beobachtungen",
        "BR24 ground truth is used only for display and evaluation, never for radar association or inference. Processed trajectories supply the position observations": "BR24-Ground-Truth wird ausschließlich für Anzeige und Evaluation verwendet, niemals für Radarzuordnung oder Inferenz. Aufbereitete Trajektorien liefern die Positionsbeobachtungen",
        "A dash means unavailable or not yet identifiable; motion and noise estimates need sufficient observations": "Ein Strich bedeutet nicht verfügbar oder noch nicht identifizierbar; Bewegungs- und Rauschschätzungen benötigen genügend Beobachtungen",
        "Position uses the selected m/km display, speed m/s, angles degrees and turn rate degrees/s; time is shown as h, min, s": "Position in der gewählten Einheit m/km, Geschwindigkeit in m/s, Winkel in Grad und Drehrate in Grad/s; Zeit in h, min, s",
        "Set plausible thresholds and tail probabilities for speed, turn rate and noise; thresholds are not hard limits": "Plausible Schwellenwerte und Überschreitungswahrscheinlichkeiten für Geschwindigkeit, Drehrate und Rauschen festlegen; Schwellenwerte sind keine harten Grenzen",
        "Minimum effective position uncertainty; distinct from additional simulated position noise in Analysis setup": "Minimale effektive Positionsunsicherheit; getrennt vom zusätzlich simulierten Positionsrauschen in den Analyseeinstellungen",
        "Sequential Monte Carlo updates a weighted particle approximation as observations arrive": "Sequenzielles Monte Carlo aktualisiert eine gewichtete Partikelapproximation mit jeder eintreffenden Beobachtung",
        "Initializes once from three accepted measurements, then updates one measurement at a time; distinct from rolling batch VI": "Initialisiert einmal aus drei akzeptierten Messungen und aktualisiert danach jeweils mit einer neuen Messung; unterscheidet sich von rollierender Batch-VI",
        "Edit settings for the selected method, such as particle count, posterior draws or convergence controls": "Einstellungen des gewählten Verfahrens bearbeiten, etwa Partikelanzahl, Posterior-Samples oder Konvergenzvorgaben",
        "Apply saves validated settings for the next analysis; Cancel or the dialog X discards unapplied edits": "Übernehmen speichert geprüfte Einstellungen für die nächste Analyse; Abbrechen oder das Fenster-X verwirft ungespeicherte Änderungen",
        "Mean 2D error of the current median forecast against available future reference positions, in meters; lower is better.": "Mittlerer 2D-Fehler der aktuellen Medianprognose gegenüber verfügbaren zukünftigen Referenzpositionen in Metern; kleiner ist besser.",
        "Position error of the median at the final forecast step in meters; lower is better.": "Positionsfehler des Medians beim letzten Prognoseschritt in Metern; kleiner ist besser.",
        "Root mean squared 2D position error of the median across all forecast steps in meters; lower is better.": "Wurzel des mittleren quadrierten 2D-Positionsfehlers des Medians über alle Prognoseschritte in Metern; kleiner ist besser.",
        "Joint 2D probabilistic forecast error; lower is better.": "Gemeinsames probabilistisches 2D-Prognosefehlermaß; kleiner ist besser.",
        "Mean negative log predictive density of future reference positions across the forecast horizon; unavailable without a forecast or future reference positions.": "Mittlere negative logarithmische prädiktive Dichte zukünftiger Referenzpositionen über den Prognosehorizont; ohne Prognose oder zukünftige Referenzpositionen nicht verfügbar.",
        "Cumulative share of evaluated future positions inside per-step 2D 50% covariance ellipses, shown as percentage and count; not whole-path coverage.": "Kumulativer Anteil ausgewerteter zukünftiger Positionen innerhalb der zeitschrittweisen 2D-50%-Kovarianzellipsen, als Prozentwert und Anzahl; keine Abdeckung vollständiger Pfade.",
        "Cumulative share of evaluated future positions inside per-step 2D 90% covariance ellipses, shown as percentage and count; not whole-path coverage.": "Kumulativer Anteil ausgewerteter zukünftiger Positionen innerhalb der zeitschrittweisen 2D-90%-Kovarianzellipsen, als Prozentwert und Anzahl; keine Abdeckung vollständiger Pfade.",
        "Wall-clock time for the current update, including forecast generation but excluding GUI drawing; not forecast accuracy.": "Reale Laufzeit des aktuellen Updates einschließlich Prognoseberechnung und ohne GUI-Zeichnung; kein Maß für die Prognosegenauigkeit.",
        "Raw BR24 tracking supports SMC, Sequential VI, VI, and MCMC and starts at index 0. Original radar timestamps are retained; set additional position noise to 0": "BR24-Rohdaten unterstützen SMC, sequenzielle VI, VI und MCMC ab Index 0. Ursprüngliche Radarzeitstempel bleiben erhalten; zusätzliches Positionsrauschen auf 0 setzen",
        "Initial measurement, Associated, or Prediction only when no detection is accepted": "Erste Messung, Zugeordnet oder Nur Prädiktion, wenn keine Detektion akzeptiert wird",
        "Original radar-cycle counter; N counts displayed updates, not detection IDs": "Ursprünglicher Radarzykluszähler; N zählt angezeigte Updates und keine Detektions-IDs",
        "Detection selected within the current radar cycle": "Innerhalb des aktuellen Radarzyklus ausgewählte Detektion",
        "Normalized innovation squared: position discrepancy relative to predicted measurement uncertainty; unavailable at initialization": "Normiertes Innovationsquadrat: Positionsabweichung relativ zur vorhergesagten Messunsicherheit; bei Initialisierung nicht verfügbar",
        "Cartesian measurement covariance entries in square meters, not posterior or forecast covariance; cov_xy may be negative": "Einträge der kartesischen Messkovarianz in Quadratmetern, nicht der Posterior- oder Prognosekovarianz; cov_xy kann negativ sein",
        "Tracking remains available; reference-state and reference-based forecast metrics show dashes": "Tracking bleibt verfügbar; Referenzzustand und referenzbasierte Prognosemetriken zeigen Striche",
        "Select layers, forecast regions, and the follow-ship view span": "Darstellungsebenen, Prognoseregionen und die Ansichtsgröße beim Verfolgen wählen",
        "Posterior for speed, heading, and turn rate": "Posterior für Geschwindigkeit, Kurswinkel und Drehrate",
        "Posterior for observation and process noise": "Posterior für Beobachtungs- und Prozessrauschen",
        "Dashed gray reference path; visibility does not affect inference or evaluation": "Gestrichelter grauer Referenzpfad; seine Sichtbarkeit beeinflusst Inferenz und Evaluation nicht",
        "Observations through N for position CSVs; Bayesian track through N for filtered BR24 positions. The legend follows the displayed data": "Beobachtungen bis N bei Positions-CSV-Dateien; Bayesscher Track bis N bei gefilterten BR24-Positionen. Die Legende richtet sich nach den angezeigten Daten",
        "Solid red coordinate-wise median of all future latent-position draws; on by default, not a single sampled trajectory": "Durchgezogener roter koordinatenweiser Median aller zukünftigen latenten Positions-Samples; standardmäßig aktiv, keine einzelne gezogene Trajektorie",
        "Dashed red mean of all future latent-position draws, independent of the displayed path count; off by default. Mean and median can be shown together": "Gestrichelter roter Mittelwert aller zukünftigen latenten Positions-Samples, unabhängig von der angezeigten Pfadanzahl; standardmäßig aus. Mittelwert und Median können gemeinsam angezeigt werden",
        "ADE, FDE and RMSE always evaluate the median, regardless of visible forecast lines; display switches do not change inference, regions or scores": "ADE, FDE und RMSE bewerten immer den Median, unabhängig von sichtbaren Prognoselinien; Anzeigeoptionen verändern Inferenz, Regionen und Metriken nicht",
        "Sampled future paths illustrating uncertainty; their display count does not change the full draws used for regions and metrics": "Gezogene zukünftige Pfade zur Darstellung der Unsicherheit; ihre Anzeigeanzahl verändert die für Regionen und Metriken genutzten vollständigen Samples nicht",
        "Off by default: paths start at their sampled current latent positions. On: replace the drawn start with the current displayed point for orientation only; forecasts and metrics stay unchanged": "Standardmäßig aus: Pfade beginnen an ihren gezogenen aktuellen latenten Positionen. Eingeschaltet wird der gezeichnete Start nur zur Orientierung durch die angezeigte aktuelle Position ersetzt; Prognosen und Metriken bleiben unverändert",
        "Apply commits plot changes without restarting inference; Cancel or X discards unapplied edits. Playback stays accessible": "Übernehmen wendet Plot-Änderungen ohne Neustart der Inferenz an; Abbrechen oder X verwirft ungespeicherte Änderungen. Die Wiedergabe bleibt zugänglich",
        "Viewport size in meters; manual pan/zoom stays available. Disable Follow ship to keep the view fixed": "Ansichtsgröße in Metern; manuelles Verschieben und Zoomen bleiben möglich. Schiff verfolgen deaktivieren, um die Ansicht festzuhalten",
        "Toggle each level independently; regions describe future latent positions, not added measurement noise": "Beide Niveaus unabhängig schalten; die Regionen beschreiben zukünftige latente Positionen und kein zusätzliches Messrauschen",
        "Filled density region pooling draws across future steps; not a simultaneous probability band for the whole path": "Gefüllte Dichteregion aus gemeinsam betrachteten Samples zukünftiger Schritte; kein simultanes Wahrscheinlichkeitsband für den gesamten Pfad",
        "Contour outlines of the pooled horizon density instead of a filled region": "Konturlinien der gepoolten Horizontdichte anstelle einer gefüllten Region",
        "Separate filled density regions for each future timestamp, preserving step-specific uncertainty": "Getrennte gefüllte Dichteregionen für jeden zukünftigen Zeitstempel mit zeitschrittspezifischer Unsicherheit",
        "Separate density contour outlines for each future timestamp": "Getrennte Dichtekonturen für jeden zukünftigen Zeitstempel",
        "Per-step 2D regions from empirical means and covariances; an elliptical approximation to the forecast draws": "Zeitschrittweise 2D-Regionen aus empirischen Mittelwerten und Kovarianzen; elliptische Approximation der Prognose-Samples",
        "Hide spatial regions without disabling forecasts, sampled paths or evaluation": "Räumliche Regionen ausblenden, ohne Prognosen, gezogene Pfade oder Evaluation zu deaktivieren",
        "Coverage always uses per-step covariance ellipses, independently of the selected visual uncertainty style": "Die Abdeckung verwendet stets zeitschrittweise Kovarianzellipsen, unabhängig von der gewählten Unsicherheitsdarstellung",
    }
)


def normalize_language(language):
    """Accept the two explicitly supported presentation languages."""
    if language not in {"en", "de"}:
        raise ValueError("Language must be 'en' or 'de'.")
    return language


def _pattern(template):
    parts = []
    for literal, field, _format_spec, _conversion in Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append(f"(?P<{field}>.*?)")
    return re.compile("".join(parts), re.DOTALL)


def translate(text, language="en"):
    """Translate labels and complete templates; keep numbers and paths intact."""
    normalize_language(language)
    text = str(text)
    source = _ENGLISH.get(text, text)
    if source in TRANSLATIONS:
        return TRANSLATIONS[source] if language == "de" else source
    for english_pattern, german_pattern, english, german in _TEMPLATES:
        match = english_pattern.fullmatch(text) or german_pattern.fullmatch(text)
        if match is not None:
            fields = match.groupdict()
            if "field" in fields:
                fields["field"] = translate(fields["field"], language)
            return (german if language == "de" else english).format_map(fields)
    if text.endswith(":"):
        return translate(text[:-1], language) + ":"
    for suffix in (" (derived)", " (abgeleitet)"):
        if text.endswith(suffix):
            return text[: -len(suffix)] + (
                " (abgeleitet)" if language == "de" else " (derived)"
            )
    return text


def widget_language(widget):
    """Resolve language from the owning application, without global state."""
    while widget is not None:
        attributes = getattr(widget, "__dict__", {})
        if "_trajectory_language" in attributes:
            return attributes["_trajectory_language"]
        widget = attributes.get("master")
    return "en"


def translate_for(widget, text):
    return translate(text, widget_language(widget))


def localize_widgets(widget, language=None):
    """Refresh display labels and dropdown captions while retaining field values."""
    if language is None:
        language = widget_language(widget)
        if language == "en":
            return
    normalize_language(language)
    pending = [widget]
    while pending:
        current = pending.pop()
        options = current.keys()
        if "text" in options:
            current.configure(text=translate(current.cget("text"), language))
            if current.winfo_class() == "Label" and int(current.cget("width")) > 0:
                original_wrap = vars(current).setdefault(
                    "_trajectory_wrap", current.cget("wraplength")
                )
                wrap = original_wrap
                if language == "de" and not int(original_wrap):
                    from tkinter import font as tkfont

                    font = tkfont.Font(root=current, font=current.cget("font"))
                    wrap = font.measure("0" * int(current.cget("width")))
                current.configure(wraplength=wrap)
        if current.winfo_class() in {"Tk", "Toplevel"}:
            current.title(translate(current.title(), language))
        if "textvariable" in options and current.winfo_class() in {"Label", "TLabel"}:
            variable = str(current.cget("textvariable"))
            if variable:
                current.setvar(variable, translate(current.getvar(variable), language))
        if "values" in options:
            choices = current.cget("values")
            selected = current.get()
            if selected in choices:
                current.configure(
                    values=tuple(translate(item, language) for item in choices)
                )
                translated = translate(selected, language)
                if translated != selected:
                    current.set(translated)
        pending.extend(current.winfo_children())


def localize_figure(figure, language):
    """Translate Matplotlib text artists without changing data or navigation."""
    for artist in figure.findobj(Text):
        text = artist.get_text()
        translated = translate(text, language)
        if translated != text:
            artist.set_text(translated)


_ENGLISH = {german: english for english, german in TRANSLATIONS.items()}
_TEMPLATES = tuple(
    (_pattern(english), _pattern(german), english, german)
    for english, german in TRANSLATIONS.items()
    if "{" in english
)
