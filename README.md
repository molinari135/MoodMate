# MoodMate
Analisi del parlato e dello scritto per la rilevazione della depressione adolescenziale
![MoodMate (1)](https://user-images.githubusercontent.com/64893048/216070880-cd7eb40e-e2d4-45d7-b513-6449a50210d3.png)

## Gruppo di lavoro
- **Giacomo Signorile**, 704897, g.signorile14@studenti.uniba.it
- **Ester Molinari**, 716555, e.molinari3@studenti.uniba.it

#### AA 2022-23

## Istruzioni per l'uso
Il programma è stato implementato in Python su Colab. Prima di eseguire il codice, scaricare la cartella filebot e caricarla nel Notebook.

![image](https://user-images.githubusercontent.com/64893048/216153218-25e2cadc-f803-4718-bb83-cc42983ce199.png)

Per eseguirlo, andare su [Colab](https://colab.research.google.com/drive/1_C3-A6j4SXfyW64WGO4qRSDk15MT1Nzv?usp=sharing) alla sezione Chatbot MoodMate ed eseguire il blocco di codice.

![image](https://user-images.githubusercontent.com/64893048/216152751-3f66b652-f2a1-405a-9afc-647cca2c7fbd.png)

Attendere qualche secondo per l'avvio del chatbot che si presenterà nel seguente modo, dopo aver aperto il menu a tendina:

![image](https://user-images.githubusercontent.com/64893048/216152946-2927e636-5a03-4336-866e-e741b37872f6.png)

Per provare la maggior parte delle funzionalità, si consiglia di eseguire il comando `voice`.

> Questa soluzione è stata adottata per evitare un problema generato dalla libreria che si occupa di trascrivere il file audio, che non si manifesta su Colab

## Obiettivi
Secondo l’Organizzazione Mondiale della Sanità (OMS), circa il 10-20% degli adolescenti in tutto il mondo soffre di disturbi mentali, tra cui la depressione. In uno studio condotto nel 2018, si stima che circa il 5-10% degli adolescenti in Italia soffra di depressione. Questa percentuale viene influenzata da molteplici fattori, tra cui la salute mentale, lo stile di vita, le circostanze familiari e la disponibilità di trattamenti adeguati.

Il nostro obiettivo è di fornire un chatbot in grado di rilevare una possibile situazione di depressione analizzando tre aspetti:

1. Costruzione delle frasi e utilizzo delle parole durante una banale conversazione non guidata affidata ad un bot
2. Analisi del tono della voce dato da un audio che comprende una breve descrizione della propria giornata
3. Test composto da 10 domande non cliniche per accertare la possibile situazione di depressione

## Target
Il nostro progetto è indirizzato per gli adolescenti (over 13) ed appare come una normale chat, analizzandone le informazioni prima di proporre un'analisi o un quiz per approfondire la condizione di tristezza dell'utente.

## Screenshots
![Immagine WhatsApp 2023-02-01 ore 19 26 29](https://user-images.githubusercontent.com/64893048/216153651-d43b8526-2001-4d85-98c4-00647e39e13f.jpg)
![Immagine WhatsApp 2023-02-01 ore 19 27 17](https://user-images.githubusercontent.com/64893048/216153676-e4cd566b-be41-44ee-82d2-a4f9250a3f9b.jpg)
![Immagine WhatsApp 2023-02-01 ore 19 27 44](https://user-images.githubusercontent.com/64893048/216153705-eb2e655d-c0f5-4be2-af76-ed709da27d80.jpg)
![Immagine WhatsApp 2023-02-01 ore 19 28 08](https://user-images.githubusercontent.com/64893048/216153776-f633b0b5-16e3-4066-85ea-c6110f3bf348.jpg)

## Problemi noti
- A volte il programma non esce nonostante venga inserito il comando `exit`
- Quando viene eseguito `voice`, il test viene ripetuto più volte



