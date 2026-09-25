# Piano di implementazione — miglioramenti da `CODE_ANALYSIS.md`

**Stato: tutte le fasi (0-4) completate.** Tre bug reali sono stati scoperti durante
l'implementazione (o in discussione successiva) e fixati subito, non solo pianificati — vedi
`KNOWN_ISSUES.md` #12 (`to_minimize` non escludeva davvero i pixel mascherati dal fit), #13
(`rmse_test` ignorava il parametro `scale` ricevuto), e #14 (il kernel Moffat interno usato durante
il fit aveva dimensione pari, introducendo un bias sistematico di ~0.5 pixel su `dx`/`dy` — e
distorcendo `fwhm`/`alpha` anche nei fit senza offset). Una deviazione dal piano originale:
`plot_results` è rimasta in `utils.py` (non spostata in `plotting.py` come le altre funzioni di
plotting) perché ricostruisce il modello del fit prima di disegnarlo, usando le stesse funzioni
interne di `to_minimize` — spostarla avrebbe creato un import circolare tra `utils.py` e
`plotting.py`.

Piano d'azione per i miglioramenti individuati in `CODE_ANALYSIS.md`, ordinato per rischio/valore.
Non tocca i bug già coperti da `KNOWN_ISSUES.md`.

**Escluso per ora**: il test sintetico di iniezione-recupero (PSF nota iniettata + verifica del
recupero). Tentativi precedenti non hanno funzionato; da riprendere più avanti come attività a sé
stante, non incluso nelle fasi sotto.

Ogni fase è pensata per essere mergeable/testabile in isolamento, in modo da poter fermarsi dopo
una qualunque di esse senza lasciare il pacchetto in uno stato rotto.

---

## Fase 0 — Rete di sicurezza

Obiettivo: avere una baseline solida prima di toccare la logica di fit, così ogni fase successiva
può essere verificata per non-regressione.

- Verificare che `tests/test_utils.py` copra già `to_minimize`/`run_measure_psf`/`rmse_test` con
  almeno un caso "happy path" con dati sintetici piccoli (non serve iniezione-recupero realistica,
  basta un caso deterministico che produca un risultato riproducibile).
- Se manca, aggiungere un test di regressione che salvi l'output attuale di `run_measure_psf` su un
  caso sintetico minimale (FWHM, eventualmente covarianza una volta disponibile) — serve da
  riferimento per confermare che le fasi 1-4 non cambiano il risultato numerico su quel caso.
- Nessuna modifica al codice di produzione in questa fase.

**Uscita**: baseline di test verde, snapshot del risultato del fit su un caso di riferimento.

---

## Fase 1 — Migrazione `leastsq` → `least_squares` con bound veri

Obiettivo: sostituire il meccanismo di penalità (`factor = 1e10`) con bound nativi e ottenere una
Jacobiana pulita da cui calcolare la covarianza dei parametri.

File: `src/musepsf/utils.py` (`to_minimize`, `run_measure_psf`, `rmse_test`).

Passi:
1. **Residui non mascherati**: modificare `to_minimize` per restituire un array pieno invece di un
   `MaskedArray` — `function.filled(0).ravel()` invece di `function.ravel()` (`apply_mask` produce
   il masked array; i pixel mascherati devono contribuire residuo zero, non essere "nascosti").
   Verificare che `leastsq` desse davvero lo stesso risultato in precedenza (i pixel mascherati non
   dovrebbero pesare sul risultato in nessuno dei due schemi).
2. **Bound dinamici**: scrivere un piccolo helper (accanto a `_unpack_fit_params`, es.
   `_pack_bounds(fit_alpha, offset, fwhm_bound, alpha_bound, dd_bound)`) che costruisca le tuple
   `(lower, upper)` della stessa lunghezza di `p0`, seguendo la stessa logica con cui
   `_unpack_fit_params` interpreta `pars` in base a `fit_alpha`/`offset`.
3. **Sostituire la chiamata**: `leastsq(...)` → `least_squares(to_minimize, x0=p0, bounds=bounds,
   args=(...), xtol=1e-9, max_nfev=600)`. Notare che `least_squares` non ha `full_output`: i
   risultati utili sono `res.x` (equivalente a `best_fit`), `res.jac`, `res.fun`, `res.cost`.
4. **Ricalcolare la covarianza**: dato `res.jac` (Jacobiana allo `x` finale) e i gradi di libertà
   `dof = res.fun.size - len(res.x)`, calcolare `cov = inv(J^T J) * (2*res.cost / dof)` (stesso
   principio di `curve_fit`/quello che `leastsq` calcolava internamente in `res[1]`). Salvare come
   nuovo attributo, es. `self.best_fit_cov`, accanto a `self.best_fit`.
5. **Adeguare `to_minimize`/`plot_results`/`rmse_test`** ai bound taggati come argomenti (oggi sono
   già passati come liste, verificare solo la coerenza di firma dopo il refactor punto 2).
6. Aggiornare `MUSEImage.measure_psf` (`musepsf.py:238`) per esporre `self.best_fit_cov` insieme a
   `self.best_fit`.

Test:
- Confrontare `best_fit` sul caso di riferimento (Fase 0) tra `leastsq` (ramo precedente) e
  `least_squares` (nuovo): deve coincidere entro una tolleranza numerica stretta.
- Test dedicato che verifichi che un fit con parametro iniziale fuori bound converga comunque
  correttamente (dove prima si affidava al fattore di penalità).
- Test che la covarianza calcolata sia semanticamente ragionevole (positiva sulla diagonale,
  dimensione coerente con `len(best_fit)`).

**Rischio**: medio — è la fase più delicata perché tocca il cuore numerico del pacchetto. Da
mergeare solo dopo che il confronto con la baseline è pulito.

---

## Fase 2 — `rmse_test` come quality control

Obiettivo: trasformare `rmse_test` da solo strumento diagnostico manuale a controllo di qualità
utilizzabile in automatico dopo ogni `measure_psf`, con due livelli di sfruttamento indipendenti
(si può implementare solo A, solo B, o entrambi).

File: `src/musepsf/utils.py` (`rmse_test`), `src/musepsf/musepsf.py` (`measure_psf`).

### Opzione A — Intervalli di confidenza dalla mappa Δχ²

- `rmse_test` già calcola `rmse_cube` su una griglia attorno al best fit. Aggiungere una funzione
  `confidence_from_rmse_cube(rmse_cube, test_yx, test_fwhm, dof)` che:
  - converte l'RMSE cube in una superficie Δχ² (Δχ² = dof · [(RMSE/RMSE_min)² − 1], assumendo che
    l'RMSE minimo sulla griglia approssimi il minimo globale già trovato dal fit);
  - per ciascun parametro (dx, dy, fwhm) isola la proiezione 1D già calcolata
    (`rmse_proj_x/y/f`) e trova gli estremi dove Δχ² supera la soglia desiderata (1 per 1σ, ~4 per
    2σ su un singolo parametro);
  - restituisce un intervallo di confidenza per parametro, da riportare accanto al best fit.
- Utile soprattutto come cross-check indipendente della covarianza ottenuta in Fase 1 (via
  Jacobiana): se le due stime divergono molto, è un segnale che il minimo non è ben approssimato da
  una forma quadratica locale (es. per bound attivi o superfici molto asimmetriche).

### Opzione B — Flag QC pass/fail automatico

- Definire un criterio semplice per marcare un fit come "sospetto" senza intervento manuale, ad
  esempio:
  - il minimo della griglia `rmse_cube` non coincide (entro una cella) con il centro della griglia
    (best fit) → il fit di `leastsq`/`least_squares` potrebbe essere caduto in un minimo locale
    diverso da quello vicino al best fit atteso;
  - la superficie è troppo piatta (variazione di RMSE tra centro e bordo griglia sotto una soglia
    relativa) → il parametro non è ben vincolato dai dati (es. troppo pochi pixel non mascherati,
    `edge` troppo aggressivo);
  - il best fit è vicino al bordo di uno dei bound (`fwhm_bound`, `dd_bound`) → coerente con quanto
    già loggato oggi ma da rendere un flag esplicito invece di solo un print.
- Il risultato è un piccolo dizionario/dataclass `QCResult(passed: bool, reasons: list[str])`
  restituito da `run_measure_psf`/esposto come `self.qc` su `MUSEImage`, non solo un plot.
- `plot` esistente di `rmse_test` resta invariato per l'ispezione visiva; l'opzione B aggiunge la
  lettura automatica della stessa griglia già calcolata, senza duplicare il calcolo.

Passi comuni:
1. Rendere `rmse_test` chiamabile senza side-effect di plotting obbligatori (separare calcolo della
   griglia dalla parte `if plot`/`_finalize_plot`, così sia A che B possono richiamarla anche con
   `plot=False` in modalità "solo QC").
2. Esporre un parametro in `MUSEImage.measure_psf` (es. `qc=False`) che, se `True`, esegue
   `rmse_test` sulla griglia attorno al best fit appena trovato e popola `self.qc_confidence`
   (opzione A) e/o `self.qc` (opzione B).
3. Documentare nel docstring di `measure_psf` che `qc=True` aggiunge un costo computazionale (la
   griglia richiede `num_points**3` valutazioni di `to_minimize`), da usare quindi come step
   opzionale, non di default per run massivi.

Test:
- Test unitario di `confidence_from_rmse_cube` su una griglia sintetica con minimo noto (es.
  parabola discretizzata), verificando che l'intervallo di confidenza recuperato corrisponda a
  quello atteso analiticamente.
- Test unitario dei criteri QC (opzione B) su griglie sintetiche costruite apposta per far scattare
  ciascuna delle condizioni (minimo spostato dal centro, superficie piatta, bound al limite).

**Rischio**: basso — è additivo, non modifica il percorso di fit esistente.

---

## Fase 3 — Refactor firma di `to_minimize`/`run_measure_psf`

Obiettivo: sostituire gli 8+ argomenti posizionali condivisi (`fxx`, `fyy`, `arrayslices`,
`starmask`, `zeromask`, `oversample`, ...) con un singolo oggetto di contesto immutabile.

File: `src/musepsf/utils.py`.

Passi:
1. Introdurre una `@dataclass FitContext` con i campi oggi passati a mano: `starmask`, `nanmask`,
   `fxx`, `fyy`, `arrayslices`, `oversample`, `scale`, e i bound (`fwhm_bound`, `alpha_bound`,
   `dd_bound`).
2. `_prepare_fit_arrays` costruisce direttamente un `FitContext` invece di restituire una tupla di
   5 elementi.
3. `to_minimize(pars, convolved, reference, ctx, alpha0=None)` — firma ridotta da 11 a 5 parametri.
4. Aggiornare `plot_results`, `run_measure_psf`, `rmse_test` di conseguenza.
5. Nessun cambio di comportamento atteso: refactor puramente strutturale.

Test: la stessa baseline della Fase 0 deve restituire risultati bit-identici (non solo "entro
tolleranza", visto che qui non cambia nessuna operazione numerica).

**Rischio**: basso, ma da fare *dopo* la Fase 1 per evitare di rifare il refactor due volte sulla
stessa firma di funzione.

---

## Fase 4 — Separazione plotting/logging

Obiettivo: rendere `utils.py` importabile e testabile senza side-effect grafici, e sostituire i
`print()` con `logging`.

File: nuovo `src/musepsf/plotting.py`; `src/musepsf/utils.py`, `image.py`, `musepsf.py`.

Passi:
1. Spostare in `plotting.py`: `get_norm`, `_finalize_plot`, `plot_images`, `plot_results`,
   `plot_psf`, e le porzioni di plotting dentro `check_flux_calibration`/`rmse_test`/
   `build_startable` (quest'ultime restano nei moduli originali ma chiamano le funzioni di plot
   importate da `plotting.py` invece di costruire le figure inline).
2. Introdurre un logger di modulo (`logging.getLogger(__name__)`) in `utils.py`, `image.py`,
   `musepsf.py`; sostituire i `print(...)` esistenti con `logger.info(...)`/`logger.warning(...)`
   a seconda del contenuto (es. `"Removing N close stars"` → info; warning già esistenti restano
   `warnings.warn`, non toccati).
3. Nessuna modifica di firma pubblica: gli import da `utils.py` in `musepsf.py`/`image.py` restano
   validi (le funzioni di plotting sono ri-esportate o gli import vengono aggiornati al nuovo
   modulo).

Test: nessun nuovo test numerico necessario; verificare solo che l'import di `musepsf` non fallisca
e che i test esistenti (che già passano `save=False, show=False`) continuino a passare senza
richiedere un display grafico.

**Rischio**: basso, puramente organizzativo.

---

## Ordine consigliato e dipendenze

```
Fase 0 (baseline)
   │
   ▼
Fase 1 (least_squares + covarianza)  ──┐
   │                                    │
   ▼                                    ▼
Fase 3 (refactor FitContext)      Fase 2 (rmse_test come QC, indipendente)
   │
   ▼
Fase 4 (plotting/logging, indipendente dalle altre)
```

Fase 2 e Fase 4 non dipendono l'una dall'altra né stretto da Fase 3, e possono procedere in
parallelo rispetto ad esse una volta completata la Fase 0. Fase 3 conviene farla dopo la Fase 1 per
non ritoccare due volte la stessa firma di `to_minimize`.

## Esplicitamente fuori scope in questo piano

- Test sintetico di iniezione-recupero end-to-end — rimandato, tentativi precedenti non riusciti.
- Propagazione dell'incertezza della calibrazione di flusso (ODR) nel fit della PSF — scartato,
  vedi discussione in `CODE_ANALYSIS.md`.
- χ² pesato per varianza del pixel — scartato per motivi di robustezza al mascheramento stellare,
  vedi `CODE_ANALYSIS.md`.
