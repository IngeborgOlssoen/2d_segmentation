
# 2D_segmentering av CTCA bilder. 


## Beskrivelse

Vårt prosjekt var å utvikle en modell for presis 3D-segmentering av CTCA-bilder. Denne modellen bruker 2D UNet fra Monai til å segmentere og predikere hvor sykdommen er lokalisert på bilet. 

## Funksjoner 
- **Datainnlasting**: Støtter import av data fra nrrd-filer
- **Datapreprossesing**: Tar inn CTCA og Annotations, og samler dem i par. Videre deles den tilfeldig inn i trening, validering og testsett, henholdsvis 60%,20% og 20% av datasettet. 
- **Analyse**: Under trening gir den output med trening-og validerings feil, og i evaluering gir den dice-og HD95 score. 
- **Visualisering**: Genererer grafiske fremstillinger av ground truth mask og predicted mask, sammen med tilhørende bilde. 

## Installasjon
Før du installerer, sørg for at du har python 3.6+ og pip installert på maskinen din. I tillegg må du lagre prosjektet på work på cybele-pcene.  Du må enten være remote eller på PC-ene på cybelelab for å kunne få tak i datasettet. 
For å kunne kjøre koden må man installere pakkene som ligger i requirements.txt. 

## Når du kjører
Kjør først data_prepros, deretter data_loader, så train og til slutt, evaluering. 

```bash
ssh user@clab[1-26].idi.ntnu.no
pip install -r requirements.txt

python /work/user/2d_segmentation/data_prepros.py
python /work/user/2d_segmentation/data_loader.py
python /work/user/2d_segmentation/train.py
python /work/user/2d_segmentation/evaluate.py









