CREATE TABLE public.dataset (
    "age" INT,
    "sex" TEXT,
    "bmi" NUMERIC(4,2),
    "children" INT,
    "smoker" TEXT,
    "region" TEXT,
    "charges" NUMERIC(16,2)
);

COPY public.dataset(
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges"
) FROM '/var/lib/postgresql/data/insurance.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE public.models (
  "modelName"       TEXT PRIMARY KEY,
  "modelType"       TEXT NOT NULL,
  "modelParams"     TEXT NOT NULL,
  "isTrained"       BOOLEAN NOT NULL DEFAULT False,
  "trainAccuracy"   NUMERIC(21,20),
  "testAccuracy"    NUMERIC(21,20),
  "weights"         BYTEA,
  "modifyDate"      TIMESTAMP NOT NULL DEFAULT now()
);
