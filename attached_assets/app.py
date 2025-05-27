import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import json

# Konfiguration av sidan
st.set_page_config(
    page_title="CPM-modell för ILS-mjukvaror - Dynamisk ROC-analys",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialisera session state variabler"""
    if 'csf_list' not in st.session_state:
        st.session_state.csf_list = [
            "Efterlevnad av ILS-ramverk",
            "Pris för kund",
            "Tidsbesparing",
            "Skalbarhet drift",
            "Informationssäkerhetsklassning",
            "Skalbarhet AI",
            "Funktionell bredd inom ILS",
            "Förmåga att tolka och hantera olika indataformat",
            "Supportkostnad",
            "Output - Struktur",
            "Grad av automation",
            "Time-to-deploy",
            "Systemintegration",
            "Robusthet",
            "Output - Filformat",
            "Användarvänlighet (UI/UX)",
            "Kundbas",
            "Utbildningsbehov",
            "Övrig funktionalitet"
        ]
    if 'vendor_list' not in st.session_state:
        st.session_state.vendor_list = ["Combitech", "Konkurrent A", "Konkurrent B"]
    if 'ratings_df' not in st.session_state:
        st.session_state.ratings_df = pd.DataFrame()
    if 'csf_order' not in st.session_state:
        st.session_state.csf_order = list(range(len(st.session_state.csf_list)))

def calculate_roc_weights(n_csfs, order):
    """Beräkna ROC-vikter (Rank Order Centroid) för givna CSF:er"""
    # Skapa en ordning där lägst värde = högst prioritet
    # order innehåller 0-indexerade positioner, vi behöver konvertera till rankning
    rank_positions = [0] * n_csfs
    for i, pos in enumerate(order):
        rank_positions[pos] = i + 1  # 1-indexerad ranking (1, 2, 3, ...)
    
    weights = []
    for i in range(n_csfs):
        rank = rank_positions[i]
        # ROC-formel: w_i = (1/N) * sum(1/k för k från rank till N)
        # Säkerställ att k aldrig är 0 genom att använda 1-indexerad ranking
        weight = (1/n_csfs) * sum(1/k for k in range(max(1, rank), n_csfs + 1))
        weights.append(weight)
    
    # Normalisera så summan blir 1.0
    total = sum(weights)
    if total > 0:
        normalized_weights = [w/total for w in weights]
    else:
        # Fallback till lika vikter om något går fel
        normalized_weights = [1/n_csfs] * n_csfs
    return normalized_weights

def get_current_csf_data():
    """Hämta aktuell CSF-data med ROC-viktning"""
    csf_names = st.session_state.csf_list
    order = st.session_state.csf_order
    weights = calculate_roc_weights(len(csf_names), order)
    
    # Skapa rankning baserat på ordning (0 = högst prioritet)
    rank_mapping = {pos: i + 1 for i, pos in enumerate(sorted(order))}
    
    return [
        {"name": csf_names[i], "weight": weights[i], "rank": rank_mapping[order[i]]}
        for i in range(len(csf_names))
    ]

def initialize_ratings_dataframe():
    """Initialisera eller uppdatera ratings DataFrame"""
    csf_data = get_current_csf_data()
    vendors = st.session_state.vendor_list
    
    # Skapa ny DataFrame med aktuella CSF:er och vendors
    new_df = pd.DataFrame(index=vendors, columns=[csf["name"] for csf in csf_data])
    
    # Fyll med befintliga värden om de finns
    if not st.session_state.ratings_df.empty:
        for vendor in vendors:
            for csf in csf_data:
                if (vendor in st.session_state.ratings_df.index and 
                    csf["name"] in st.session_state.ratings_df.columns):
                    new_df.loc[vendor, csf["name"]] = st.session_state.ratings_df.loc[vendor, csf["name"]]
    
    # Fyll tomma celler med 3 (standardbetyg)
    new_df = new_df.fillna(3).infer_objects(copy=False)
    st.session_state.ratings_df = new_df

def calculate_results():
    """Beräkna viktade resultat baserat på aktuella betyg och ROC-vikter"""
    if st.session_state.ratings_df.empty:
        return pd.DataFrame()
    
    csf_data = get_current_csf_data()
    vendors = st.session_state.vendor_list
    
    results = []
    for vendor in vendors:
        raw_sum = 0
        weighted_sum = 0
        
        for csf in csf_data:
            if csf["name"] in st.session_state.ratings_df.columns:
                rating = st.session_state.ratings_df.loc[vendor, csf["name"]]
                raw_sum += rating
                weighted_sum += rating * csf["weight"]
        
        # Normaliserad poäng (0-100 skala)
        normalized_score = (weighted_sum / 4.0) * 100
        
        results.append({
            "Vendor": vendor,
            "Raw Sum": round(raw_sum, 2),
            "Weighted Sum": round(weighted_sum, 3),
            "Normalized (0-100)": round(normalized_score, 1)
        })
    
    return pd.DataFrame(results)

def create_bar_chart(results_df):
    """Skapa stapeldiagram över viktade resultat"""
    if results_df.empty:
        return None
    
    fig = px.bar(
        results_df, 
        x="Vendor", 
        y="Weighted Sum",
        title="Viktade totalpoäng per leverantör",
        color="Weighted Sum",
        color_continuous_scale="Viridis",
        text="Weighted Sum"
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis=dict(range=[0, 4]))
    
    return fig

def create_heatmap():
    """Skapa heatmap över alla betyg"""
    if st.session_state.ratings_df.empty:
        return None
    
    # Konvertera DataFrame till numerisk för heatmap
    df_numeric = st.session_state.ratings_df.astype(float)
    
    fig = go.Figure(data=go.Heatmap(
        z=df_numeric.values,
        x=df_numeric.columns,
        y=df_numeric.index,
        colorscale='RdYlGn',
        zmin=1,
        zmax=4,
        text=df_numeric.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Poäng")
    ))
    
    fig.update_layout(
        title="Heatmap: Alla betyg per CSF och leverantör",
        xaxis_title="Kritiska Framgångsfaktorer (CSF)",
        yaxis_title="Leverantörer", 
        height=400
    )
    
    return fig

def export_to_csv(results_df, csf_data):
    """Exportera alla data till CSV"""
    output = io.StringIO()
    
    # Exportera CSF-viktningar
    output.write("=== CSF VIKTNINGAR (ROC-metod) ===\n")
    csf_df = pd.DataFrame(csf_data)
    csf_df.to_csv(output, index=False, sep=';')
    
    output.write("\n=== BETYG PER LEVERANTÖR OCH CSF ===\n")
    st.session_state.ratings_df.to_csv(output, sep=';')
    
    output.write("\n=== SAMMANFATTADE RESULTAT ===\n")
    results_df.to_csv(output, index=False, sep=';')
    
    return output.getvalue()

def main():
    initialize_session_state()
    
    st.title("📊 CPM-modell för ILS-mjukvaror")
    st.markdown("### Dynamisk ROC-analys med anpassningsbara CSF:er och leverantörer")
    
    st.markdown("""
    **Denna applikation erbjuder:**
    - Dynamiska CSF:er (lägg till/ta bort kritiska framgångsfaktorer)
    - Automatisk ROC-viktberäkning (Rank Order Centroid)
    - Flexibel leverantörslista
    - Realtidsuppdatering av resultat
    - Export till CSV för vidare analys
    """)
    
    # Sidebar för konfiguration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # CSF-hantering
        st.subheader("📋 Kritiska Framgångsfaktorer (CSF)")
        
        # Lägg till ny CSF
        new_csf = st.text_input("Lägg till ny CSF:")
        if st.button("➕ Lägg till CSF") and new_csf and new_csf not in st.session_state.csf_list:
            st.session_state.csf_list.append(new_csf)
            st.session_state.csf_order.append(len(st.session_state.csf_list) - 1)
            initialize_ratings_dataframe()
            st.success(f"Lade till: {new_csf}")
            st.rerun()
        
        # Ta bort CSF
        if len(st.session_state.csf_list) > 1:
            csf_to_remove = st.selectbox("Ta bort CSF:", [""] + st.session_state.csf_list)
            if st.button("🗑️ Ta bort CSF") and csf_to_remove:
                idx = st.session_state.csf_list.index(csf_to_remove)
                st.session_state.csf_list.remove(csf_to_remove)
                st.session_state.csf_order = [i if i < idx else i-1 for i in st.session_state.csf_order if i != idx]
                initialize_ratings_dataframe()
                st.success(f"Tog bort: {csf_to_remove}")
                st.rerun()
        
        # CSF-prioritering för ROC-vikter
        st.subheader("🎯 CSF-prioritering (för ROC-vikter)")
        st.write("Dra för att ändra prioritetsordning:")
        
        # Visa aktuell ordning och låt användaren ändra
        csf_data = get_current_csf_data()
        sorted_csfs = sorted(csf_data, key=lambda x: x["rank"])
        
        for i, csf in enumerate(sorted_csfs):
            new_rank = st.number_input(
                f"{csf['name'][:30]}...",
                min_value=1,
                max_value=len(st.session_state.csf_list),
                value=csf["rank"],
                key=f"rank_{csf['name']}"
            )
            if new_rank != csf["rank"]:
                # Uppdatera ordning
                csf_idx = st.session_state.csf_list.index(csf["name"])
                st.session_state.csf_order[csf_idx] = new_rank - 1
                st.rerun()
        
        st.divider()
        
        # Leverantörshantering
        st.subheader("🏢 Leverantörer")
        
        # Lägg till ny leverantör
        new_vendor = st.text_input("Lägg till ny leverantör:")
        if st.button("➕ Lägg till leverantör") and new_vendor and new_vendor not in st.session_state.vendor_list:
            st.session_state.vendor_list.append(new_vendor)
            initialize_ratings_dataframe()
            st.success(f"Lade till: {new_vendor}")
            st.rerun()
        
        # Ta bort leverantör
        if len(st.session_state.vendor_list) > 1:
            vendor_to_remove = st.selectbox("Ta bort leverantör:", [""] + st.session_state.vendor_list)
            if st.button("🗑️ Ta bort leverantör") and vendor_to_remove:
                st.session_state.vendor_list.remove(vendor_to_remove)
                initialize_ratings_dataframe()
                st.success(f"Tog bort: {vendor_to_remove}")
                st.rerun()
    
    # Initialisera ratings DataFrame
    initialize_ratings_dataframe()
    
    # Visa aktuella CSF-vikter
    st.header("📊 Aktuella CSF-vikter (ROC-metod)")
    csf_data = get_current_csf_data()
    csf_df = pd.DataFrame(csf_data)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(csf_df, use_container_width=True)
    with col2:
        # Viktfördelning som cirkeldiagram
        fig_pie = px.pie(
            csf_df, 
            values='weight', 
            names='name', 
            title="Viktfördelning"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Poängsättning
    st.header("🎯 Poängsättning")
    st.markdown("**Sätt betyg för varje leverantör och CSF (1 = Mycket dålig, 4 = Mycket bra)**")
    
    # Organisera poängsättning i kolumner
    cols = st.columns(len(st.session_state.vendor_list))
    
    for col_idx, vendor in enumerate(st.session_state.vendor_list):
        with cols[col_idx]:
            st.subheader(f"🏢 {vendor}")
            
            for csf in csf_data:
                current_rating = st.session_state.ratings_df.loc[vendor, csf["name"]]
                
                # Skapa kort CSF-namn för bättre layout
                short_name = csf['name'][:25] + "..." if len(csf['name']) > 25 else csf['name']
                st.write(f"**{short_name}**")
                st.caption(f"Vikt: {csf['weight']:.3f}")
                
                # Knappar för poängsättning
                button_cols = st.columns(4)
                rating_labels = ["1️⃣ Mycket dålig", "2️⃣ Dålig", "3️⃣ Bra", "4️⃣ Mycket bra"]
                
                new_rating = current_rating
                for i, (btn_col, label) in enumerate(zip(button_cols, rating_labels)):
                    with btn_col:
                        rating_value = i + 1
                        button_type = "primary" if current_rating == rating_value else "secondary"
                        
                        if st.button(
                            f"{rating_value}",
                            key=f"{vendor}_{csf['name']}_{rating_value}",
                            type=button_type,
                            help=label,
                            use_container_width=True
                        ):
                            new_rating = rating_value
                
                # Uppdatera DataFrame om värdet ändrats
                if new_rating != current_rating:
                    st.session_state.ratings_df.loc[vendor, csf["name"]] = new_rating
                    st.rerun()
                
                st.divider()
    
    # Beräkna och visa resultat
    st.header("📈 Resultat")
    results_df = calculate_results()
    
    if not results_df.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Sammanfattning")
            st.dataframe(results_df, use_container_width=True)
        
        with col2:
            st.subheader("📈 Viktade poäng")
            bar_chart = create_bar_chart(results_df)
            if bar_chart:
                st.plotly_chart(bar_chart, use_container_width=True)
        
        # Heatmap
        st.subheader("🎨 Detaljerad heatmap")
        heatmap = create_heatmap()
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
        
        # Export
        st.subheader("💾 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_to_csv(results_df, csf_data)
            st.download_button(
                label="📄 Ladda ner som CSV",
                data=csv_data,
                file_name="cpm_roc_analys.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_data = {
                "csf_data": csf_data,
                "vendors": st.session_state.vendor_list,
                "ratings": st.session_state.ratings_df.to_dict(),
                "results": results_df.to_dict(),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            st.download_button(
                label="📋 Ladda ner som JSON",
                data=json.dumps(json_data, ensure_ascii=False, indent=2),
                file_name="cpm_roc_analys.json",
                mime="application/json"
            )
    
    # Metodologi
    with st.expander("📚 ROC-metodologi", expanded=False):
        st.markdown("""
        ### Rank Order Centroid (ROC) Viktberäkning
        
        ROC-metoden beräknar vikter baserat på rangordning av CSF:er:
        
        **Formel:** w_i = (1/N) × Σ(k=i to N) 1/k
        
        där:
        - w_i = vikt för CSF på rang i
        - N = totalt antal CSF:er
        - Summan normaliseras så att Σw_i = 1.0
        
        **Fördelar med ROC:**
        - Automatisk viktberäkning baserad på prioritetsordning
        - Matematiskt väldefinierad och reproducerbar
        - Enkel att förstå och justera
        
        **Poängskala:**
        - 1 = Mycket dålig prestanda
        - 2 = Dålig prestanda  
        - 3 = Bra prestanda
        - 4 = Mycket bra prestanda
        """)

if __name__ == "__main__":
    main()