 #!/bin/bash
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Grocery_and_Gourmet_Food.json.gz
wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Grocery_and_Gourmet_Food.json.gz
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz

gunzip Grocery_and_Gourmet_Food.json.gz
gunzip meta_Grocery_and_Gourmet_Food.json.gz
gunzip Grocery_and_Gourmet_Food_5.json.gz
