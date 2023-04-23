(ns ptonner.pytorch-clj
  (:require [camel-snake-kebab.core :as csk]
            [clojure.tools.logging :as log]
            [clojure.string :as str]
            clojure.walk
            [libpython-clj2.python :as py :refer
             [cfn path->py-obj python-type py. py..]]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python.class :as py-class]
            [libpython-clj2.python.np-array]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.tensor :as dst]
            tech.v3.datatype.casting
            [tech.v3.datatype.errors :as errors]))

(py/initialize!)
(defonce torch-module (py/import-module "torch"))
(log/info "'pytorch' version found: "
          (get (py/module-dict torch-module) "__version__"))

(require-python '[torch])

(defn tensor->ndarray [tensor] (py. tensor numpy :force true))

(defn ndarray->tensor [nd] (torch/from_numpy nd))

;!zprint {:format :skip}
(comment
  (-> (torch/zeros 10)
      tensor->ndarray))

(defn ndarray->ds
  [pyobject]
  (-> pyobject
      py/->jvm
      dst/tensor->dataset))

(defn tensor->ds
  [tensor]
  (-> tensor
      tensor->ndarray
      ndarray->ds))

;!zprint {:format :skip}
(comment
  (-> (torch/zeros 10 10)
      tensor->ds))

(defn- assert-numeric-ds
  [ds]
  (errors/when-not-error
   (every? true?
           (map #(-> %
                     meta
                     :datatype
                     tech.v3.datatype.casting/numeric-type?)
                (ds/columns ds)))
   "All values of target columns need to be numeric."))

(defn numeric-ds->ndarray
  [ds]
  (assert-numeric-ds ds)
  (-> ds
      dst/dataset->tensor
      py/->python))

(defn numeric-ds->tensor
  [ds]
  (-> ds
      numeric-ds->ndarray
      ndarray->tensor))

;!zprint {:format :skip}
(comment
  (-> (torch/ones 10 10)
      tensor->ds
      numeric-ds->tensor))

(require-python '[torch.utils.data :refer [TensorDataset]])

(def split-ds (juxt cf/feature cf/target))

(defn ds->tensor-dataset
  [ds]
  (->> (split-ds ds)
       (map numeric-ds->tensor)
       (apply cfn TensorDataset)))

;!zprint {:format :skip}
(comment
  (-> (torch/ones 10 10)
      tensor->ds
      (ds-mod/set-inference-target [0 1])
      ds->tensor-dataset
      ;; ((juxt cf/feature cf/target))
      ;; (#(map numeric-ds->tensor %))
      ;; (#(apply cfn TensorDataset %))
      (py/get-item 0)))

(require-python '[torch.nn :as nn])

(defn module
  [name & args]
  (let [mod (path->py-obj
             (str "torch.nn."
                  (if (keyword? name) (csk/->PascalCaseString name) name)))]
    (apply cfn mod args)))

(defn clone
  [n module-fn & args]
  (apply cfn nn/Sequential (repeatedly n #(apply module-fn args))))

(def ^:private -Module
  (py/create-class
   "-Module"
   [nn/Module]
   {"__init__" (py-class/make-tuple-instance-fn
                (fn [self fwd modules]
                  (py. nn/Module __init__ self)
                  (py/set-attrs! self
                                 {"fwd" fwd, "modules" (nn/ModuleList modules)})
                  nil)),
    "forward" (py-class/make-tuple-instance-fn
               (fn [self & args]
                 (let [fwd (py/get-attr self "fwd")] (apply cfn fwd args))))}))

(defn as-module
  [m]
  (cond (and (= :pyobject (type m)) (py/is-instance? m nn/Module)) m
        (fn? m) (as-module (m))
        (sequential? m) (cfn nn/ModuleList (map as-module m))
        :else m))

;!zprint {:format :skip}
(comment
  (as-module (nn/Linear 3 10))
  (as-module #(nn/Linear 3 10))
  (as-module [(nn/Linear 3 10) (nn/Linear 4 10)])
  (as-module (repeatedly 2 #(nn/Linear 3 10))))

(defn add
  [& modules]
  (let [modules (map as-module modules)
        bcast (apply juxt modules)
        fwd (fn [X] (reduce torch/add (bcast X)))]
    (-Module fwd modules)))

;!zprint {:format :skip}
(comment
  ;; NOTE: this seems to show the overhead is pretty bad (e.g. twice
  ;; as slow)? although maybe it would be comparable going into a pure
  ;; python class as well, since a certain level of cost comes from
  ;; getting the python attrs in the forward method? should check this
  (let [l1 (nn/Linear 3 4)
        l2 (nn/Linear 3 4)
        x (torch/randn 2 3)
        bcast (juxt l1 l2)
        fwd (fn [X] (apply cfn torch/add (bcast X)))
        a (add l1 l2)]
    (time (a x))
    (time (fwd x))
    (time (torch/add (l1 x)
                     (l2 x))))
  (let [mods (repeatedly 100 #(nn/Linear 3 3))
        x (torch/randn 2 3)
        bcast (apply juxt mods)
        fwd (fn [X] (reduce torch/add (bcast X)))
        a (apply add mods)
        s (apply cfn nn/Sequential mods)]
    (time (a x))
    (time (fwd x))
    (time (s x)))
  (reduce torch/add
          [(torch/randn 2 3)
           (torch/randn 2 3)
           (torch/randn 2 3)])
  ((apply juxt [(nn/Linear 3 10)
                (nn/Linear 3 10)])
   (torch/randn 2 3))
  (let [a (add (nn/Linear 3 10)
               (nn/Linear 3 10))
        x (torch/randn 2 3)]
    (a x)))

(defn chain [& modules] (apply cfn nn/Sequential (map as-module modules)))

;!zprint {:format :skip}
(comment

  (as-module (repeatedly 2 #(apply cfn nn/Linear [10 10])))

  (py/is-instance? (repeatedly 2 #(nn/Linear 10 10)) nn/Module)

  (apply add (repeatedly 2 #(nn/Linear 10 10)))
  (add (repeatedly 2 #(nn/Linear 10 10)))


  (let [mods (repeatedly 3 #(nn/Linear 10 10))
        X (torch/randn 3 10)
        broadcast (apply juxt mods)]
    (reduce torch/add (broadcast X))
    ;; (apply cfn torch/add
    ;;        (broadcast X))
    )

  (clone 10 nn/Linear 4 4)
  (clone 10 (fn [& args] (apply cfn  nn/Linear args)) 10 10)
  (let [m (nn/Sequential
           (add (nn/Linear 3 10) (nn/Linear 3 10))
           ;; (clone 3 nn/Linear 3 10)
           (nn/Linear 10 20 :bias false))]
    (m (torch/randn 6 3)))

  
  (def ffl)
  (chain
   (ffl )))
