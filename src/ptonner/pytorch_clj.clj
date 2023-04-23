(ns ptonner.pytorch-clj
  (:require [camel-snake-kebab.core :as csk]
            [clojure.tools.logging :as log]
            [clojure.string :as str]
            clojure.walk
            [libpython-clj2.python :as py :refer
             [cfn path->py-obj python-type py. py.. py.-
              with-gil-stack-rc-context]]
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

(def pyobject? #(= :pyobject (type %)))
(defn resolve-py
  [module obj]
  (if (pyobject? obj)
    obj
    (let [name (if (keyword? obj) (csk/->PascalCaseString obj) (name obj))
          module (if (str/ends-with? module ".") module (str module "."))]
      (path->py-obj (str module name)))))

;; Datasets

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

(defn maybe->tensor-dataset
  [ds]
  (cond (and (pyobject? ds) (= (py/python-type ds) :tensor-dataset)) ds
        (and (pyobject? ds) (= (py/python-type ds) :tensor)) (TensorDataset ds)
        (ds/dataset? ds) (ds->tensor-dataset ds)
        :else ds))

;!zprint {:format :skip}
(comment
  (py/python-type (torch/ones 10 10))
  (-> (torch/ones 10 10)
      tensor->ds
      ;; (ds-mod/set-inference-target [0 1])
      ;; ds->tensor-dataset
      ;; py/python-type
      type
      ;; (py/get-item 0)
      ))

;; Modules

(require-python '[torch.nn :as nn])
(defonce torch-nn-module (py/import-module "torch.nn"))

(defn module
  [name & args]
  (let [mod (path->py-obj
             (str "torch.nn."
                  (if (keyword? name) (csk/->PascalCaseString name) name)))]
    (apply cfn mod args)))

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
        (sequential? m) (map as-module m)
        :else m))

(defn module? [m] (py/is-instance? m (py/get-attr torch-nn-module "Module")))


;!zprint {:format :skip}
(comment
  ;; (require-python '[torch.nn.functional :as nnf])
  (as-module (nn/Linear 3 10))
  (as-module #(nn/Linear 3 10))
  (as-module [(nn/Linear 3 10) (nn/Linear 4 10)])
  (as-module (repeatedly 2 #(nn/Linear 3 10))))

(defn add
  [& modules]
  (let [modules (flatten (as-module modules))
        bcast (apply juxt modules)
        fwd (fn [X] (reduce torch/add (bcast X)))]
    (-Module fwd modules)))

;!zprint {:format :skip}
(comment
  (as-module [(repeatedly 2 #(nn/Linear 3 4))])
  (as-module [(nn/Linear 3 4) (nn/Linear 3 4)])
  ;; NOTE: basic tests
  (let [l1 (nn/Linear 3 4)
        l2 (nn/Linear 3 4)
        l3 (nn/Linear 3 4)
        x (torch/randn 2 3)
        a (add l1 l2 l3)]
    (torch/allclose (a x) (reduce torch/add
                                  [(l1 x) (l2 x) (l3 x)])))
  (let [mods (repeatedly 10 #(nn/Linear 3 4))
        x (torch/randn 2 3)
        a (add mods)]
    (torch/allclose (a x) (reduce torch/add
                                  (map #(% x) mods))))
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
    (time (s x))))

(defn clone
  [n module-fn & args]
  (apply cfn nn/Sequential (repeatedly n #(apply module-fn args))))

;!zprint {:format :skip}
(comment
  (let [c (clone 3 nn/Linear 3 3)
        mods (python/list c)
        x (torch/randn 3 3)]
    (torch/allclose
     (c x)
     (reduce #(%2 %1) x mods))))

(defn chain
  [& modules]
  (->> modules
       (map as-module)
       flatten
       (apply cfn nn/Sequential)))


;!zprint {:format :skip}
(comment
  (let [c1 (nn/Linear 3 4)
        c2 (add (repeatedly 3 #(nn/Linear 4 2)))
        c (chain c1 c2)
        x (torch/randn 2 3)]
    (torch/allclose
     (c x)
     (-> x
         c1
         c2)))
  (let [c1 (nn/Linear 3 4)
        c2 (add (repeatedly 3 #(nn/Linear 4 2)))
        c3 (clone 3 nn/Linear 2 2)
        c (chain c1 c2 c3)
        x (torch/randn 2 3)]
    (torch/allclose
     (c x)
     (-> x c1 c2 c3))))

;; Optimizer

(defn maybe->parameters
  [model-or-parameters]
  (cond (and (pyobject? model-or-parameters) (module? model-or-parameters))
        (py. model-or-parameters parameters)
        (sequential? model-or-parameters) (flatten (map maybe->parameters
                                                        model-or-parameters))
        :else model-or-parameters))

(defn optim-group
  ([model-or-parameters] (optim-group model-or-parameters {}))
  ([model-or-parameters options]
   (let [params (maybe->parameters model-or-parameters)]
     (py/->py-dict (assoc options "params" params)))))

;!zprint {:format :skip}
(comment
  (maybe->parameters (nn/Linear 10 10))
  (maybe->parameters (py. (nn/Linear 10 10) parameters))
  (py/is-instance? (optim-group (nn/Linear 10 10) {:lr 1e-3})
                   python/dict))

(defn- optim-name
  "Need to do some specialized work with optimizer names b/c of atypical naming conventions (e.g. SGD)"
  [o]
  (-> o
      name
      csk/->PascalCaseString
      (str/replace #"(?i)sgd" "SGD")
      (str/replace #"(?i)rms" "RMS")
      (str/replace #"(?i)bfgs" "BFGS")))

(comment (resolve-py "torch.optim" (optim-name :adam))
         (resolve-py "torch.optim" (optim-name :adam-w))
         (resolve-py "torch.optim" (optim-name :sgd))
         (resolve-py "torch.optim" (optim-name :asgd))
         (resolve-py "torch.optim" (optim-name :lbfgs))
         (resolve-py "torch.optim" (optim-name :rmsprop)))


(defn as-optim
  [parameters {:keys [algorithm], :as kw}]
  (let [params (maybe->parameters parameters)
        optim (resolve-py "torch.optim" (optim-name algorithm))]
    (apply cfn optim params (dissoc kw :algorithm))))

;; Training

(require-python '[torch.utils.data :refer [DataLoader]])

(defn- minibatch
  [model loss-fn optimizer device X Y]
  (with-gil-stack-rc-context (py. optimizer zero_grad)
                             (let [X (py. X to device)
                                   Y (py. Y to device)
                                   Yhat (model X)
                                   loss (loss-fn Yhat Y)]
                               (py. loss backward)
                               (py. optimizer step))))

(defn train
  ([dataset model loss-fn] (train dataset model loss-fn {}))
  ([dataset model loss-fn
    {:keys [optimizer loader parameters epochs device],
     :or {optimizer {:algorithm :adam-w}, loader {}, epochs 1}}]
   (let [dataset (maybe->tensor-dataset dataset)
         loader (apply cfn DataLoader dataset loader)
         parameters (or parameters [(optim-group model)])
         optimizer (as-optim parameters optimizer)
         device (or device "cpu")]
     (py. model train)
     (dorun
      (for [e (range epochs)]
        (dorun (for [batch loader]
                 (apply minibatch model loss-fn optimizer device batch)))))
     model)))

;!zprint {:format :skip}
(comment
  (let [p 10
        N 1000
        truth (nn/Linear p 1)
        X (torch/randn N p)
        y (py/with [_ (torch/no_grad)] (truth X))
        y (torch/add y (torch/mul 0.1 (torch/randn_like y)))
        ds (TensorDataset X y)
        trained (train ds (nn/Linear p 1)
                       (resolve-py "torch.nn.functional" "mse_loss")
                       {:epochs 10
                        ;; :loader {:batch_size 32}
                        })]
    (py/with [_ (torch/no_grad)]
             (-> (torch/sub
                  (py.- truth weight)
                  (py.- trained weight))
                 torch/abs
                 torch/sum))))

(let [p 10
      N 100
      truth (nn/Linear p 1)
      X (torch/randn N p)
      y (py/with [_ (torch/no_grad)] (truth X))
      y (torch/add y (torch/mul 0.1 (torch/randn_like y)))
      ds (TensorDataset X y)]
  (train ds
         (nn/Linear p 1)
         (resolve-py "torch.nn.functional" "mse_loss")
         {:epochs 10}))
