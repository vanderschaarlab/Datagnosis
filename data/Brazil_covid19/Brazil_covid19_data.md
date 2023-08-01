# Dataset Reference
Ministry of Health. SIVEP-Gripe public dataset. 2020.
http://plataforma.saude.gov.br/coronavirus/dados-abertos/
(accessed May 10, 2020; in Portuguese).

# Pre-processing

## Normalising column values

### Region

Mapping: SG_UF_NOT &rarr; Region

|SG_UF_NOT|Region|
|---------|------|
|DF|Central-West|
|GO|Central-West|
|MS|Central-West|
|MT|Central-West|
|AM|North|
|PA|North|
|RR|North|
|TO|North|
|AL|Northeast|
|BA|Northeast|
|CE|Northeast|
|MA|Northeast|
|PB|Northeast|
|PE|Northeast|
|PE|Northeast|
|PI|Northeast|
|RN|Northeast|
|SE|Northeast|
|PR|South|
|RS|South|
|SC|South|
|ES|Southeast|
|MG|Southeast|
|RJ|Southeast|
|SP|Southeast|

### Ethnicity

Mapping: Race &rarr; Ethnicity
|Race|Ethnicity|
|----|---------|
|Parda|Mixed|
|Branca|White|
|Preta|Black|
|Amarela|East Asian|
|Indigena|Indigenous|

## Encoding

|Sex|Num|
|---|---|
|F|0|
|M|1|


|Ethnicity|Num|
|---------|---|
|Mixed|0|
|White|1|
|Black|2|
|East Asian|3|
|Indigenous|4|

|Region|Num|
|------|------|
|Central-West|0|
|North|1|
|Northeast|2|
|South|3|
|Southeast|4|
