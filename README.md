# ML-Project: Gestures

Der Use-Case für dieses Projekt ist eine digitale Steuerung einer Powerpointpräsentation mithilfe von Gesten. Dazu nimmt die Kamera die Handbewegungen auf und ein zweistufiges Modell findet die Gesten.

Das Script wurde in _Python 3.12_ implementiert.

## Teammitglieder

|Name|Matrikelnummer|
|-|-|
|Joel Bück|4860895|
|Lukas Runge|7590014|
|Martin Schauer|7961802|
|Lukas Stamm|8402366|

# Projektaufbau

```
ML_Projekt
├── etc
│    └── utils.py
├── models
│    ├── classification
│    │    └── ...
│    └── mediapipe
│         └── ...
├── training
│    └── skeleton_time_series
│         └── ...
├── Ausarbeitung_Projektbericht.pdf
├── main.py
├── model_training.py
└── ...
```

Sowohl das Trainingsscript ```model_training.py``` als auch das Anwendungsscript ```main.py``` benötigen einige Funktionen, die aus Lesbarkeitsgründen nach ```utils.py``` ausgelagert wurden.

## ML-Bibliotheken

Es werden 2 Modelle benötigt: Eines zur Handdetektion und eines zur Zeitserienklassifikation.

Die Handdetektion wird mit einem vortrainierten Modell von Google durchgeführt. Dazu wird die ```mediapipe``` Bibliothek genutzt.

Das Klassifikationsmodell wurde selbst trainiert und nutzt die ```pytorch``` Bibliothek.

## Traningsscript

```model_training.py``` 

Im Traningsscript werden zunächst die Modellarchitekur für das Klassifikationsmodell und der Dataloader implementiert. Danach ist ein Standard Trainings- und Validierungs-Loop implementiert.

### Performance

Mediapipe ermöglicht eine relativ stabile Framerate von ca. 10 fps.

Das Zeitreihenklassifikationsmodell ist sehr lightweigt und hat fast keine implikation für die Framerate. Hier sind die anderen Performance-Kennzahlen:

|Stat||
|-|-|
|Cross-Entropy-Loss|0.32|
|Accuracy|85%|
|Recall|85%|
|Precision|87%|

## Anwendung

```main.py```

Hier werden zunächst eine Powerpoint-integration, die Modelle und die Integration der Kamera geladen.

Danach wird in einer Endlos-Schleife das aktuelle Frame der Kamera gelesen und:

- Ein Puffer gefüllt, um die korrekte Tensor-shape sicherzustellen
- Die Zeitreihe klassifiziert
    - Falls ein positives Ergebnis gefunden wird, wird ein dementsprechender Befehl in Powerpoint ausgeführt und sämtlliche Gestenerkennung für eine festgelegte Zeit blockiert (Cooldown)
    - Falls kein positives Ergebnis gefunden wird, passiert nichts