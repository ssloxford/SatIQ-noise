#!/bin/bash

db_path=data
storage_path=$db_path/noise

NOISE=0
NOISE_MAX=32
NOISE_INCREMENT=7
COUNTER=0

NOISE=$NOISE docker-compose up -d

while true; do
  # Wait 15 minutes
  sleep 900

  docker-compose down

  mv "$db_path/db.sqlite3" "$db_path/db-temp.sqlite3"

  NOISE_OLD=$NOISE
  NOISE=$((NOISE+NOISE_INCREMENT))
  while [ $NOISE -gt $NOISE_MAX ]; do
    ((NOISE -= (NOISE_MAX + 1)))
  done
  NOISE=$NOISE docker-compose up -d

  mkdir -p "$storage_path/$NOISE_OLD"
  while [ -f "$storage_path/$NOISE_OLD/db-$COUNTER.sqlite3" ]; do
    COUNTER=$((COUNTER+1))
  done
  mv "$db_path/db-temp.sqlite3" "$storage_path/$NOISE_OLD/db-$COUNTER.sqlite3"
done
