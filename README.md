# DepthMapEstimation

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Inhalt</summary>
  <ol>
    <li>
      <a href="#anleitung-zur-nutzung-des-programms">Anleitung zur Nutzung des Programms</a>
    </li>
    <li><a href="#referenzen">Referenzen</a></li>
    <li><a href="#verwendete-module">Verwendete Module</a></li>
  </ol>
</details>

# Anleitung zur Nutzung des Programms 
[Zum Programm "DepthMapEstimation"](DepthMapEstimation.py)

Das Programm beginnt damit immer abwechselnd die zu Vergleichenden Bilder, Feature Points, Matches, Epipolarlinien, rectifizierten Bilder und die Disparitätskarte anzuzeigen. Das ganze geschieht für alle Kombinationen aus dem ersten Bild und einem der anderen angelegten Bilder (im Ordner Data). Nachdem entsprechend alle Bilder durchlaufen wurden, wird die gemittelte Tiefenkarte aus allen Disparitätskarten berechnet und dargestellt.

# Referenzen
[Depth Map from Stereo Images OpenCv](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
[Easily Create a Depth Map with Smartphone AR (Part 1 - 4)](https://www.andreasjakl.com/easily-create-depth-maps-with-smartphone-ar-part-1/)

# Verwendete Module
|Modul          |Version    |
|---------------|-----------|
|numpy          |1.22.3     |
|opencv-python  |4.5.5.64   |
