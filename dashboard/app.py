import streamlit as st
import numpy as np
import sys
import os

# --- Pfad zu src hinzufügen ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_model import load_model_and_data
from src.fairness_metrics import group_accuracy
from src.bias_analysis import compute_bias
from src.report_generator import generate_report

# NEU: aktualisierte Text-Bias-Funktionen
from src.text_bias_analysis import run_text_bias_test, GROUP_PROMPTS

import pandas as pd
import altair as alt
import matplotlib.pyplot as plt


# --- Dashboard Titel ---
st.title("AI Ethics Skill-Profiler Dashboard")
st.write("Analyse von Fairness & Bias in einem TensorFlow-Modell")


# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Klassifikations-Bias",
    "Text-Bias (LSTM)",
    "Bild-Bias Kategorien",
    "Bild-Bias Analyse",
    "NLP Bias Scanner",
    "Edge AI Robustness & Bias Simulator",
    "AI Ethics Live Monitor"
])




# ================================================================
# TAB 1: Klassifikations-Fairness (unverändert)
# ================================================================
with tab1:

    st.header("Klassifikations-Bias Analyse")

    if st.button("🔍 Analyse starten"):

        # 1) Modell + Daten laden
        model, X_test, y_test, groups = load_model_and_data()

        # 2) Vorhersagen
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # 3) Gruppen-Genauigkeit
        group_acc = group_accuracy(y_test, y_pred, groups)

        # 4) Bias Score
        bias = compute_bias(group_acc)

        # 5) Report generieren
        report = generate_report(group_acc, bias)

        # --- Anzeige im Dashboard ---
        st.subheader("Ergebnisse")

        st.write("### Gruppen-Genauigkeiten")
        st.json(group_acc)

        # ======================================================
        # 🚦 Ampel-Barchart
        # ======================================================
        df = pd.DataFrame({
            "Gruppe": list(group_acc.keys()),
            "Accuracy": list(group_acc.values())
        })

        def accuracy_color(acc):
            if acc < 0.5:
                return "red"
            elif acc < 0.8:
                return "orange"
            return "green"

        df["Color"] = df["Accuracy"].apply(accuracy_color)

        chart = (
            alt.Chart(df)
            .mark_bar(size=50)
            .encode(
                x=alt.X("Gruppe:N", title="Gruppe"),
                y=alt.Y("Accuracy:Q", title="Genauigkeit", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Color:N", scale=None, legend=None)
            )
        )

        st.write("### Gruppen-Genauigkeit (Diagramm)")
        st.altair_chart(chart, use_container_width=True)
        # ======================================================

        st.write(f"### Bias Score: **{bias:.2f}**")

        if bias > 0.10:
            st.error("⚠️ Das Modell ist möglicherweise unfair.")
        else:
            st.success("✔️ Das Modell wirkt fair.")

        st.write("### Report")
        st.code(report)

    else:
        st.info("Klicke auf 'Analyse starten', um die Fairness zu prüfen.")



# ================================================================
# TAB 2: TEXT-BIAS (LSTM) – NEU, ERWEITERT & KORRIGIERT
# ================================================================
with tab2:

    st.header("Text-Bias Analyse (LSTM)")
    st.write("Das Modell generiert Texte zu verschiedenen Gruppen. "
             "Wir analysieren systematisch Sentiment & Toxicity.")


    # --- Parameter für die Analyse ---
    n_samples = st.slider("Texte pro Gruppe", 3, 20, 5)
    max_words = st.slider("Max. Wortanzahl pro Text", 5, 50, 20)


    if st.button("🔍 Text-Bias Analyse starten"):

        with st.spinner("Generiere Texte und analysiere Bias..."):
            results = run_text_bias_test(
                n_samples_per_group=n_samples,
                words_per_sample=max_words
            )

        group_results = results["group_results"]
        bias_scores = results["bias"]


        # -----------------------------------------------------
        # Übersichtstabelle
        # -----------------------------------------------------
        st.subheader("Ergebnisse pro Gruppe")

        table_data = []
        for group, data in group_results.items():
            table_data.append({
                "Gruppe": group,
                "Sentiment (Ø)": data["sentiment_avg"],
                "Toxicity (Ø)": data["toxicity_avg"],
                "Texte": data["n_texts"]
            })

        df_bias = pd.DataFrame(table_data)
        st.dataframe(df_bias, use_container_width=True)



        # -----------------------------------------------------
        # Diagramm: Sentiment
        # -----------------------------------------------------
        st.write("### Sentiment-Vergleich zwischen Gruppen")

        fig1, ax1 = plt.subplots()
        ax1.bar(df_bias["Gruppe"], df_bias["Sentiment (Ø)"])
        ax1.set_xlabel("Gruppe")
        ax1.set_ylabel("Durchschnittliches Sentiment")
        st.pyplot(fig1)



        # -----------------------------------------------------
        # Diagramm: Toxicity
        # -----------------------------------------------------
        st.write("### Toxicity-Vergleich zwischen Gruppen")

        fig2, ax2 = plt.subplots()
        ax2.bar(df_bias["Gruppe"], df_bias["Toxicity (Ø)"])
        ax2.set_xlabel("Gruppe")
        ax2.set_ylabel("Durchschnittliche Toxicity")
        st.pyplot(fig2)



        # -----------------------------------------------------
        # Bias Score Anzeige
        # -----------------------------------------------------
        st.subheader("Bias-Score")

        st.write(f"**Bias Score:** {bias_scores['bias_score']:.2f}")
        st.write(f"- Sentiment-Gap: {bias_scores['sentiment_gap']:.2f}")
        st.write(f"- Toxicity-Gap: {bias_scores['toxicity_gap']:.2f}")

        if bias_scores["bias_score"] > 0.20:
            st.error("⚠️ Hinweis: Das Modell zeigt deutliche Unterschiede zwischen Gruppen.")
        else:
            st.success("✔️ Geringe Unterschiede zwischen den Gruppen.")



        # -----------------------------------------------------
        # Beispieltexte
        # -----------------------------------------------------
        st.subheader("Beispieltexte aus den Gruppen")

        for group, data in group_results.items():
            st.markdown(f"### {group}")

            with st.expander("🔎 Beispieltexte anzeigen"):
                for t in data["examples"]:
                    st.markdown(f"> {t}")

            st.markdown("---")

    else:
        st.info("Klicke auf 'Text-Bias Analyse starten', um die Gruppen zu vergleichen.")
# ================================================================
# TAB 3: BILD-BIAS KATEGORIEN (KONZEPT)
# ================================================================
with tab3:
    st.header("Bild-Bias Kategorien")
    st.write(
        "Dieses Modul definiert die Kategorien, nach denen KI-generierte Bilder "
        "im Hinblick auf mögliche Verzerrungen analysiert werden sollen. "
        "Die Kategorien sind ethisch und datenschutzsensibel formuliert."
    )

    categories_data = [
        {
            "Kategorie": "Gender Expression",
            "Beschreibung": "Sichtbare geschlechtliche Erscheinung basierend auf äußeren Merkmalen.",
            "Messbare Ausprägungen": "maskulin erscheinend · feminin erscheinend · uneindeutig",
            "Ethischer Hinweis": "Keine Aussagen über Identität oder Zugehörigkeit, Trans-/Nonbinary-Identität nicht erkennbar."
        },
        {
            "Kategorie": "Skin Tone Cluster",
            "Beschreibung": "Helligkeits- bzw. Farbcluster der Hauttöne.",
            "Messbare Ausprägungen": "hell · mittel · dunkel",
            "Ethischer Hinweis": "Kein Rückschluss auf Ethnie, Nationalität oder 'Rasse'."
        },
        {
            "Kategorie": "Age Appearance",
            "Beschreibung": "Alterseindruck basierend auf visuellen Merkmalen.",
            "Messbare Ausprägungen": "Kind · Jugendliche Person · Erwachsene Person · Ältere Person",
            "Ethischer Hinweis": "Kein Rückschluss auf tatsächliches Alter."
        },
        {
            "Kategorie": "Visible Assistive Attributes",
            "Beschreibung": "Sichtbare Hinweise auf Hilfsmittel oder Barrieren.",
            "Messbare Ausprägungen": "Hilfsmittel sichtbar (z. B. Rollstuhl, Prothese) · keine Hilfsmittel sichtbar · unklar",
            "Ethischer Hinweis": "Keine medizinischen Diagnosen oder Aussagen über Behinderungsgrad."
        },
        {
            "Kategorie": "Clothing Style / Role Indicators",
            "Beschreibung": "Kleidungstypen, die soziale Rollen oder Berufe repräsentieren können.",
            "Messbare Ausprägungen": "Business-Outfit · Freizeitkleidung · Sportbekleidung · Uniform/berufsspezifisch",
            "Ethischer Hinweis": "Es werden nur sichtbare Muster betrachtet, keine Stereotype festgeschrieben."
        },
        {
            "Kategorie": "Visible Religious Symbols",
            "Beschreibung": "Sichtbare Kleidungsstücke oder Accessoires mit religiöser Funktion.",
            "Messbare Ausprägungen": "Kopftuch · Kippa · Kreuzanhänger · Turban · keine sichtbaren Symbole · unklar",
            "Ethischer Hinweis": "Es geht nur um sichtbare Objekte, nicht um religiöse Zugehörigkeit."
        },
        {
            "Kategorie": "Body Shape Appearance",
            "Beschreibung": "Wahrgenommene Körperform basierend auf visuellen Merkmalen.",
            "Messbare Ausprägungen": "schlank · durchschnittlich · kräftig · plus-size erscheinend · unklar",
            "Ethischer Hinweis": "Keine Bewertung, nur neutrale Beschreibung der Bilddarstellung."
        },
    ]

    df_categories = pd.DataFrame(categories_data)

    st.subheader("Kategorie-Übersicht")
    st.dataframe(df_categories, use_container_width=True)

    st.markdown("---")
    st.info(
        "Hinweis: Diese Kategorien beschreiben ausschließlich sichtbare Bildmerkmale. "
        "Es werden keine sensiblen personenbezogenen Daten im rechtlichen Sinne "
        "oder Identitäten im Hintergrund 'erraten', sondern nur visuelle Muster analysiert."
    )
# ================================================================
# TAB 4: BILD-BIAS ANALYSE
# ================================================================
with tab4:
    st.header("Bild-Bias Analyse")

    from src.image_bias_analysis import analyze_image_bias

    uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

    category = st.selectbox(
    "Kategorie auswählen",
    [
        "Skin Tone Cluster",
        "Gender Expression",
        "Age Appearance",
        "Clothing Style",
        "Visible Religious Symbols",
        "Body Shape Appearance"
    ]
    )


    if uploaded and st.button("Analyse starten"):
        st.image(uploaded, caption="Hochgeladenes Bild", width=300)

        results = analyze_image_bias(uploaded, category)

        st.subheader("Ergebnis")
        st.write(f"**Kategorie:** {results['category']}")
        st.write(f"**Erkannt als:** {results['result']}")
        st.write(f"**Konfidenz:** {results['confidence']:.2f}")

        st.markdown("### Erklärung")
        st.info(results["explanation"])

        st.markdown("### Ethik-Hinweis")
        st.warning(results["ethical_note"])

        # ================================================================
# TAB 5: NLP BIAS SCANNER
# ================================================================
with tab5:
    from src.nlp_bias_detector import analyze_text_bias, CATEGORY_DESCRIPTIONS
    import pandas as pd

    st.header("NLP Bias Scanner")
    st.write(
        "Dieses Modul prüft Texte auf potenziell diskriminierende oder "
        "ausschließende Formulierungen (z. B. in Stellenanzeigen, "
        "Kommunikation oder Webtexten)."
    )

    example_text = (
        "Wir sind ein junges, dynamisches Team und suchen eine Sekretärin. "
        "Deutsch als Muttersprache ist erforderlich. "
        "Bitte nur Bewerbungen von deutschen Staatsbürgern ohne Migrationshintergrund."
    )

    text_input = st.text_area(
        "Text zur Analyse",
        value=example_text,
        height=200,
    )

    if st.button("🔎 Text auf Bias prüfen"):
        if not text_input.strip():
            st.warning("Bitte gib einen Text ein.")
        else:
            results = analyze_text_bias(text_input)

            st.subheader("Gesamtbewertung")
            score = results["overall_score"]
            st.write(f"**Gesamt-Bias-Score:** {score:.2f} (0 = unauffällig, 1 = stark auffällig)")
            st.write(f"Anzahl gefundener kritischer Stellen: **{results['total_hits']}**")

            if score == 0:
                st.success("Keine der hinterlegten problematischen Formulierungen wurde gefunden.")
            elif score < 0.4:
                st.info("Einige problematische Formulierungen. Eine Überarbeitung ist empfehlenswert.")
            else:
                st.error("Deutliche Hinweise auf diskriminierende oder ausschließende Sprache.")

            # Übersicht pro Kategorie
            st.subheader("Kategorienübersicht")

            table_rows = []
            for cat, data in results["categories"].items():
                table_rows.append({
                    "Kategorie": cat,
                    "Beschreibung": CATEGORY_DESCRIPTIONS.get(cat, ""),
                    "Anzahl Treffer": data["n_hits"],
                })
            df_cat = pd.DataFrame(table_rows)
            st.dataframe(df_cat, use_container_width=True)

            # Detailansicht pro Kategorie
            st.subheader("Details nach Kategorie")

            for cat, data in results["categories"].items():
                if data["n_hits"] == 0:
                    continue

                with st.expander(f"{cat} – {data['n_hits']} Treffer"):
                    for hit in data["hits"]:
                        st.markdown(f"- **Gefunden:** „{hit['match']}“")
                        st.markdown(f"  - Erklärung: {hit['explanation']}")
                        st.markdown(f"  - Vorschlag: _{hit['suggestion']}_")
    else:
        st.info("Gib einen Text ein oder nutze das Beispiel und klicke auf „Text auf Bias prüfen“.")        

# ================================================================
# TAB 6: EDGE AI ROBUSTNESS & BIAS SIMULATOR
# ================================================================
with tab6:
    st.header("Edge AI Robustness & Bias Simulator")
    st.write(
        "Dieser Simulator untersucht, wie sich Edge-KI-Bedingungen wie "
        "schlechte Beleuchtung und Rauschen auf die Erkennungsqualität und "
        "potenzielle Bias-Effekte auswirken."
    )

    from src.edge_simulator import run_edge_simulations
    from src.edge_bias_analysis import analyze_edge_bias

    uploaded_edge = st.file_uploader(
        "Bild hochladen (für Edge-Simulationen)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_edge:
        st.subheader("Originalbild")
        st.image(uploaded_edge, width=300)

        if st.button("🔎 Edge-Simulationen ausführen"):
            st.info("Simuliere Edge-KI-Bedingungen…")

            # ---- EDGE SIMULATION AUSFÜHREN ----
            results = run_edge_simulations(uploaded_edge)

            # 🌓 Low Light anzeigen
            st.subheader("Low-Light Simulation (schlechte Beleuchtung)")
            st.image(
                results["low_light"],
                width=300,
                caption="Low-Light Version"
            )

            # 🌫 Noise anzeigen
            st.subheader("Noise Simulation (Sensorrauschen)")
            st.image(
                results["noise"],
                width=300,
                caption="Rauschen / Sensorausfall"
            )

            # ================================
            # FAIRNESS- & ROBUSTHEITSANALYSE
            # ================================
            st.subheader("⚖️ Fairness- & Robustheitsanalyse")

            bias_results = analyze_edge_bias(
                results["original"],
                results["low_light"],
                results["noise"]
            )

            # Scores
            low_light_score = bias_results["low_light"]["bias_impact"]
            noise_score = bias_results["noise"]["bias_impact"]

            st.write("### 📊 Bias Impact Scores")
            st.write(f"**Low Light:** {low_light_score:.2f}")
            st.write(f"**Noise:** {noise_score:.2f}")

            # Ampelsystem
            def score_color(score):
                if score < 0.2:
                    return "🟢 niedrig"
                elif score < 0.5:
                    return "🟡 mittel"
                else:
                    return "🔴 hoch"

            st.write("### 📉 Bewertung der Risiken")
            st.write(f"**Low-Light Risiko:** {score_color(low_light_score)}")
            st.write(f"**Noise Risiko:** {score_color(noise_score)}")

            # Kurzbericht
            st.subheader("📄 Kurzbericht")
            st.info(
                f"- Low-Light Bedingungen führen zu einem Bias-Impact von **{low_light_score:.2f}**.\n"
                f"- Noise Bedingungen führen zu einem Bias-Impact von **{noise_score:.2f}**.\n"
                f"- Höhere Werte zeigen, dass das Modell unter Edge-Bedingungen weniger zuverlässig und potenziell unfair arbeitet."
            )

    else:
        st.info("Bitte lade ein Bild hoch, um die Edge-Simulationen zu starten.")

# ================================================================
# TAB 7: AI Ethics Live Monitor (Webanalyse + Textanalyse)
# ================================================================
with tab7:
    st.header("AI Ethics Live Monitor (Webanalyse)")
    st.write(
        "Dieses Modul lädt den Text einer Webseite oder eines Eingabetextes "
        "und analysiert ihn hinsichtlich Toxicity (Moderation API) "
        "und Bias (GPT-Analyse)."
    )

    from src.web_text_extractor import get_clean_text_from_url
    from src.api_text_analyzer import analyze_moderation_long_text, analyze_bias_gpt

    # --- Auswahl: URL oder manueller Text
    mode = st.radio(
        "Analysemodus auswählen:",
        ["Webseite (URL)", "Manueller Text"],
        horizontal=True
    )

    text = ""

    if mode == "Webseite (URL)":
        url = st.text_input("Webseite zur Analyse (URL eingeben)")
        if st.button("🌐 Webseite laden"):
            if url:
                try:
                    text = get_clean_text_from_url(url)
                    st.session_state.web_text_monitor = text
                    st.success("Webseite erfolgreich geladen.")
                except Exception as e:
                    st.error(f"Fehler beim Laden der Webseite: {e}")
            else:
                st.warning("Bitte gib eine gültige URL ein.")

        # Text aus Session holen, falls schon geladen
        text = st.session_state.get("web_text_monitor", "")

    else:
        # manueller Text
        text = st.text_area(
            "Text zur Analyse eingeben:",
            value=st.session_state.get("manual_text_monitor", ""),
            height=200
        )
        st.session_state.manual_text_monitor = text

    if text:
        st.subheader("Extrahierter / zu analysierender Text")
        st.text_area("Text", value=text, height=200)

        if st.button("🧪 Bias & Toxicity Analyse starten"):
            with st.spinner("Analysiere Text mit OpenAI Moderation & GPT…"):

                # 1) Moderation-Analyse
                mod_results = analyze_moderation_long_text(text)

                # 2) GPT-Bias-Analyse
                gpt_results = analyze_bias_gpt(text)

            # -------------------------
            # Anzeige: Moderation API
            # -------------------------
            # --- Moderation API Ergebnisse anzeigen ---
            st.subheader("📊 Moderation API – Safety Scores")

            st.write(f"Blöcke analysiert: {mod_results['blocks']}")

            scores = mod_results["scores"]

            for category, vals in scores.items():
                st.write(
                f"**{category}:** "
                f"Avg = {vals['avg']:.3f}, "
                f"Max = {vals['max']:.3f}"
            )


            # -------------------------
            # Anzeige: GPT-Bias-Analyse
            # -------------------------
            st.subheader("🧠 GPT Bias-Analyse")

            overall = gpt_results.get("overall_risk", "unknown")
            st.write(f"**Gesamtrisiko (Bias):** {overall}")

            dims = gpt_results.get("dimensions", {})

            for dim_name, dim_data in dims.items():
                risk = dim_data.get("risk", "unknown")
                examples = dim_data.get("examples", [])
                st.write(f"**{dim_name.capitalize()}** – Risiko: {risk}")
                if examples:
                    for ex in examples:
                        st.write(f"• {ex}")

            comments = gpt_results.get("comments", [])
            if comments:
                st.subheader("📝 Kommentare")
                for c in comments:
                    st.write(f"- {c}")
    else:
        st.info("Bitte eine Webseite laden oder Text eingeben, um die Analyse zu starten.")
