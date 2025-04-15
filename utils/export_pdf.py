from fpdf import FPDF
import os

class AtletiPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_title("Informe Temporada 2024/25")
        self.set_author("Dirección Deportiva Atleti")

    def header(self):
        if os.path.exists("assets/images/logos/atm.png"):
            self.image("assets/images/logos/atm.png", x=10, y=8, w=15)
        self.set_font("Arial", "B", 14)
        self.set_text_color(39, 46, 97)  # Azul Atleti
        self.cell(0, 10, "Club Atlético de Madrid", ln=True, align="C")
        self.set_font("Arial", "", 12)
        self.cell(0, 10, "Departamento de Dirección Deportiva", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, "Informe confidencial - Temporada actual | Página %s" % self.page_no(), 0, 0, "C")

def export_to_pdf(nombre_pagina, contenido, autor="Dirección Deportiva", output_path="informe.pdf"):
    pdf = AtletiPDF()

    # Subtítulo del informe
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(39, 46, 97)
    pdf.cell(0, 10, nombre_pagina, ln=True, align="L")

    # Autoría
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Elaborado por: {autor}", ln=True)
    pdf.ln(5)

    # Contenido (multi-línea)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)

    for parrafo in contenido:
        pdf.multi_cell(0, 10, parrafo)
        pdf.ln(1)

    # Guardar archivo
    pdf.output(output_path)

def dataframe_a_pdf_contenido(df, columnas):
    contenido = []
    for _, row in df[columnas].iterrows():
        fila = [f"{col}: {row[col]}" for col in columnas]
        contenido.extend(fila)
        contenido.append("")  # Espacio entre jugadores
    return contenido
