(ns user
  (:require [libpython-clj2.python :as py]))

(py/initialize! :python-executable ".venv/bin/python")
