# Purchase prediction using delivered app logs
*Authors:*
* Mateusz Szczepanowski
* Albert Ściseł


*TASK:* "It would be good to know if a given user session will end with a purchase, thanks to which our consultants will be able to watch these sessions more closely and solve potential problems faster." We were using Polish to do this project.

* *data_processing.ipynb file:* this file contains an analysis of the provided data, incl. correlation between attributes, occurring gaps/errors etc.

* *building_model.ipynb file:* this file contains the steps to build models. The first model is based on a neural network and the second is a naive model that makes predictions based on session duration.

* *microservice:* simple implementation of microservice, which serves a prediction whether a given session will end with a purchase. To run microservice first run the following command: *pip install -r requirements.txt.*Then type: *python main.py*.

Details are included infdile *documentation.pdf*. 
**_NOTE:_**  The documentation is written in Polish.

