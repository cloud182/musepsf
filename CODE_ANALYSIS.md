# Analisi del core code di `musepsf`

Analisi focalizzata sui tre moduli core del pacchetto: [`image.py`](src/musepsf/image.py),
[`musepsf.py`](src/musepsf/musepsf.py) e [`utils.py`](src/musepsf/utils.py). Per i bug puntuali
già individuati e corretti in un refactoring precedente si veda `KNOWN_ISSUES.md`, non riproposti
qui.

## Architettura

`Image` (`image.py`) è la classe base: I/O FITS, resampling (via `reproject` o MPDAF), query Gaia,
costruzione ePSF con `photutils`. `MUSEImage` (`musepsf.py`) eredita da `Image` e implementa il
metodo principale del pacchetto, `measure_psf`, che stima la FWHM (e opzionalmente α, offset)
della PSF MUSE per cross-convoluzione con un'immagine di riferimento a risoluzione nota. `utils.py`
contiene tutta la logica numerica pesante (fit, FFT, plotting) come funzioni libere.

## Punti di forza

- **Metodo scientificamente valido**: la tecnica di cross-convoluzione (convolvi MUSE con la PSF
  nota di riferimento, convolvi il riferimento con un modello Moffat di prova, minimizza la
  differenza) è concettualmente solida e ben nota in letteratura per questo tipo di calibrazione.
- **Buona separazione dei concern**: `_prepare_fit_arrays` e `_unpack_fit_params` isolano
  correttamente il codice condiviso tra `run_measure_psf` e `rmse_test`, riducendo la
  duplicazione.
- **Gestione robusta dei casi limite geometrici**: dilatazione della maschera ai bordi, gestione
  di `NoOverlapError` nei cutout, `binary_fill_holes` per buchi di resampling — dettagli da chi ha
  già debuggato problemi reali di edge-effect.
- **Diagnostica estesa**: quasi ogni fase produce plot (rescaling, stelle localizzate, PSF/residui,
  RMSE map). Fondamentale in un pacchetto scientifico dove il risultato numerico da solo non basta
  a fidarsi del fit.
- **`rmse_test`** è già un embrione di analisi di sensibilità/incertezza (mappa 3D di RMSE attorno
  al best fit su fwhm/dx/dy) — buona base da cui partire per il punto sulla "confidenza" più sotto.
- **Refactoring recente ha già ripulito bug reali** (vedi `KNOWN_ISSUES.md`): mutazioni in-place
  nascoste, `assert` per validazione, unità non definite, ecc. Segno di un codice che sta maturando.

## Debolezze

### Architetturali

- `to_minimize` e `run_measure_psf` hanno **8-10 parametri posizionali** ciascuno, molti dei quali
  stato pre-calcolato (`fxx`, `fyy`, `arrayslices`) che andrebbe incapsulato in un piccolo
  oggetto/dataclass (es. `FitContext`) invece di essere passato a mano ad ogni chiamata — fragile
  e facile da disallineare.
- `utils.py` (1000+ righe) mischia responsabilità molto diverse: fit statistico, FFT, download
  SDSS via rete, plotting matplotlib. Vale la pena separare almeno `plotting.py` da `fitting.py` e
  da `sdss.py` — non solo per leggibilità, ma perché testare la logica di fit oggi richiede
  importare anche matplotlib e wget.
- `MUSEImage.measure_psf` (`musepsf.py:119`) fa troppo in un solo metodo: validazione, resampling,
  rescaling flusso, mascheramento stelle, rotazione PSF, oversampling, fit — oltre 100 righe con
  molti side-effect su `self`. Difficile da testare in isolamento o da riusare pezzo per pezzo.
- Uso di `**kwargs` "a cascata" (`kwargs.pop(...)` ripetuto) per passare parametri come `edge`,
  `fwhm0`, `mask_stars` rende la firma effettiva della funzione invisibile — bisogna leggere il
  corpo per sapere cosa si può passare.

### Correttezza / robustezza numerica

- Il fit usa `scipy.optimize.leastsq` con **bound "finti"**: se un parametro esce dai limiti,
  viene resettato a un valore arbitrario e si applica un fattore di penalità `1e10`
  (`utils.py:319-336`). Questo produce un gradiente discontinuo e può creare falsi minimi vicino ai
  bordi del bound. `scipy.optimize.least_squares` (che supporta bound nativi con Trust Region
  Reflective) sarebbe più robusto e restituirebbe anche una Jacobiana pulita — nel passaggio va
  però tenuto conto che `to_minimize` oggi restituisce un `numpy.ma.MaskedArray`
  (`apply_mask` → `convolved - ref_masked`), che `leastsq` accetta senza controlli ma su cui
  `least_squares` fallisce (`np.isfinite(f0)` interno può trovare `nan`/`inf` sotto la maschera);
  serve restituire un array "pieno" (es. `function.filled(0).ravel()`). Il vettore parametri ha
  inoltre lunghezza variabile (1-4 a seconda di `fit_alpha`/`offset`), quindi anche i bound
  andrebbero costruiti dinamicamente con la stessa logica di `_unpack_fit_params`.
- Il fit è ai **minimi quadrati non pesati**, ed è una scelta deliberata e corretta: pesare per
  l'inverso della varianza darebbe più peso proprio ai pixel ad alto SNR, cioè alle sorgenti
  puntiformi brillanti — l'esatto contrario di quello che la maschera stellare cerca di escludere.
  Se una stella sfugge al masking automatico, un fit pesato ne amplificherebbe l'effetto sul
  risultato invece di attutirlo.
- `reconstruct_psf` per SDSS ha bug noti e non fixati (issue #6 in `KNOWN_ISSUES.md`) — quel
  percorso non è validato contro dati reali.

### Qualità del codice

- Codice morto/commentato lasciato nei sorgenti (es. `# import sys; sys.exit()` in
  `utils.py:344`, commenti tipo "maybe it could be removed?" in `utils.py:501`).
- Print statement diffusi per il logging invece di `logging` — impossibile silenziare/filtrare in
  produzione o in un run batch su molte galassie.
- Nomi fuorvianti: `rmse_proj_yf`/`rmse_proj_xf` commentati come "e rot" ma non c'è nessuna
  rotazione in gioco lì (probabilmente refuso da una versione precedente del codice).
- Copertura test recente e buona (`tests/test_image.py`, `tests/test_utils.py`) ma manca ancora un
  test end-to-end sul percorso critico `measure_psf` con dati sintetici a FWHM nota, che sarebbe il
  test di validazione scientifica più importante da avere.

## Miglioramenti concreti suggeriti

1. Sostituire `leastsq` + bound finti con `least_squares(..., bounds=...)`: bound veri, e in
   output una Jacobiana da cui calcolare la matrice di covarianza dei parametri stimati. Richiede
   di far restituire a `to_minimize` un array pieno (non masked) e di costruire i bound
   dinamicamente in base alla lunghezza effettiva del vettore parametri.
2. Rifattorizzare `to_minimize`/`run_measure_psf` per passare un singolo oggetto di contesto invece
   di 8+ argomenti posizionali.
3. Spostare tutto il plotting in un modulo separato, e passare a `logging` al posto di `print`.
4. Aggiungere un test sintetico end-to-end: genera un'immagine con PSF Moffat nota, verifica che
   `measure_psf` recuperi la FWHM iniettata entro una tolleranza attesa.

## Come quantificare la confidenza nei risultati

Oggi il pacchetto restituisce solo una stima puntuale (`best_fit`), senza errori. Alcune strade,
dalla più semplice alla più completa:

- **Gratis dal fit stesso**: passando a `least_squares` con pesi corretti, la matrice di
  covarianza `(JᵀJ)⁻¹ · σ²_residuo` dà errori formali su FWHM (e α, offset se fittati) quasi senza
  costo aggiuntivo — è il primo passo naturale, visto che `leastsq` con `full_output=True` già
  restituisce `res[1]` (l'output di covarianza) ma **non viene mai letto**
  (`musepsf.py:238`, `self.best_fit = self.res[0]` — `self.res[1]` è lì e inutilizzato).
- **Sfruttare `rmse_test`**: è già una mappa di sensibilità 3D attorno al best fit. Basterebbe
  interpretarla come una superficie di verosimiglianza (Δχ² rispetto al minimo) per ricavare
  intervalli di confidenza formali sui 3 parametri, invece di limitarsi a un plot diagnostico.
- **Bootstrap/jackknife spaziale**: rifittare la PSF su sottoregioni disgiunte del campo (o
  rimuovendo a rotazione blocchi dell'immagine) e guardare la dispersione della FWHM recuperata —
  stima empirica dell'incertezza che cattura anche eterogeneità del campo (es. variazioni di
  seeing residue) che un fit gaussiano dell'errore non vede.
- **Validazione incrociata tra immagini di riferimento**: se per la stessa galassia sono
  disponibili più riferimenti (es. WFI e SDSS), confrontare la FWHM misurata con ciascuno è un
  test di accuratezza sistematica indipendente dal rumore statistico del singolo fit.
- **Test di iniezione-recupero sintetico**: iniettare PSF Moffat note su immagini realistiche
  (rumore + fondo reali) e misurare bias e varianza della FWHM recuperata su un set di casi — è lo
  standard per stabilire l'accuratezza di un metodo di questo tipo, e diventerebbe anche il test
  end-to-end di cui sopra.

Il primo intervento a più alto rapporto beneficio/sforzo è probabilmente il primo: la covarianza è
già calcolata da `leastsq` e semplicemente scartata.
