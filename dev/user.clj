(ns user
  (:require [libpython-clj2.python :as py]))

(py/initialize! :python-executable ".venv/bin/python")

;!zprint {:format :skip}
(comment
  (require '[vlaaad.reveal :as r])
  (r/tap-log))
