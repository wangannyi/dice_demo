"""Generate a dimensioned vector print master and independently detectable PNG."""
from pathlib import Path
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent

def main():
    out = ROOT / 'output'
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits = cv2.aruco.drawMarker(dictionary, 40, 6)
    png = np.full((800, 800), 255, np.uint8)
    png[100:700, 100:700] = cv2.resize(bits, (600, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / 'aruco_40.png'), png)
    pdf = canvas.Canvas(str(out / 'pdf' / 'aruco_40_30mm_A4.pdf'), pagesize=A4)
    pdf.setTitle('Robot hand marker - DICT_4X4_50 ID 40 - 30 mm')
    pdf.setFont('Helvetica-Bold', 18)
    pdf.drawString(25*mm, 267*mm, 'Robot hand marker / ID 40')
    pdf.setFont('Helvetica', 11)
    for y, line in [(253, 'Dictionary: DICT_4X4_50'), (244, 'Print at 100% / Actual size. Disable Fit to page.'),
                    (235, 'Black outer square: 30 x 30 mm. White margin: 5 mm.'),
                    (226, 'Keep the whole 40 x 40 mm white tile when cutting.')]:
        pdf.drawString(25*mm, y*mm, line)
    x, y = 85*mm, 155*mm
    pdf.setStrokeColorRGB(.7, .7, .7)
    pdf.setDash(2, 2)
    pdf.rect(x, y, 40*mm, 40*mm)
    pdf.setDash()
    pdf.setFillColorRGB(0, 0, 0)
    for row in range(6):
        for col in range(6):
            if bits[row, col] == 0:
                pdf.rect(x+(5+col*5)*mm, y+(5+(5-row)*5)*mm, 5*mm, 5*mm, stroke=0, fill=1)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.line(80*mm, 130*mm, 130*mm, 130*mm)
    for xx in (80, 130):
        pdf.line(xx*mm, 128*mm, xx*mm, 132*mm)
    pdf.drawCentredString(105*mm, 122*mm, 'This line must measure 50 mm after printing')
    pdf.drawString(25*mm, 100*mm, 'Mount flat on a rigid hand-back or wrist surface, facing the camera.')
    pdf.drawString(25*mm, 91*mm, 'Do not wrap around a curved surface or attach to moving fingers.')
    pdf.drawString(25*mm, 82*mm, 'Marker-to-palm offset must be measured after mounting.')
    pdf.save()
    _, ids, _ = cv2.aruco.detectMarkers(png, dictionary)
    assert ids is not None and ids.ravel().tolist() == [40]
    print(out / 'pdf' / 'aruco_40_30mm_A4.pdf')

if __name__ == '__main__':
    main()
