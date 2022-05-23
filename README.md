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

Das Programm muss lediglich gestartet werden und beginnt damit, immer abwechselnd die zu Vergleichenden Bilder, die Feature Points, die Matches der Feature Points,die  Epipolarlinien, die rektifizierten Bilder und die Disparitätskarte anzuzeigen. 

Zum Start des Programms werden alle zu verwendenden Bilder in ein Array geladen. Dabei dient das Bild im ersten Eintrag als Referenz ("img1" im Programm), mit welcher die übrigen Bilder im Array paarweise nacheinander verarbeitet werden. 
Hierbei werden zunächst in jedem Bild, der jeweiligen Bildpaare, die Feature-Punkte ermittelt. Diese werden anschließend, den entsprechenden Punkten im jeweils anderen Bild, gepaart. Daraufhin wird die Fundamentalmatrix, sowie eine Maske der gepaarten Punkte ermittelt. Anschließend werden mit Hilfe dieser Punkte die Epipolarlinien bzw. die beiden Epipole berechnet. 	
Mit Hilfe der Fundamentalmatrix werden dann die jeweiligen Bildpaare rektifiziert und anschließend wird die Disparität berechnet. Zuletzt werden die einzelnen Disparitäten, der zuvor verarbeiteten Bildpaare, addiert und die gemittelte Tiefenkarte errechnet.


# Referenzen
[Depth Map from Stereo Images OpenCv](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)

[Easily Create a Depth Map with Smartphone AR (Part 1 - 4)](https://www.andreasjakl.com/easily-create-depth-maps-with-smartphone-ar-part-1/)

# Verwendete Module
|Modul          |Version    |
|---------------|-----------|
|numpy          |1.22.3     |
|opencv-python  |4.5.5.64   |
