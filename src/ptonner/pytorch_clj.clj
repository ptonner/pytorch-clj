(ns ptonner.pytorch-clj
  (:require [camel-snake-kebab.core :as csk]
            [clojure.tools.logging :as log]
            [clojure.string :as str]
            clojure.walk
            [libpython-clj2.python :as py :refer
             [cfn path->py-obj python-type py. py.. py.- py* py**
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
(defonce torch-version (get (py/module-dict torch-module) "__version__"))
(log/info "'pytorch' version found: " torch-version)

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

(def ^:private -Module
  (py/create-class
   "-Module"
   [nn/Module]
   {"__init__" (py-class/make-tuple-instance-fn
                (fn module-init
                  ([self fwd] (module-init self fwd nil nil))
                  ([self fwd modules] (module-init self fwd modules nil))
                  ([self fwd modules parameters]
                   (py. nn/Module __init__ self)
                   (py/set-attrs!
                    self
                    {"fwd" fwd,
                     ;; TODO: make sure these branches line up with the types
                     ;; that will actually come in from maps and vectors
                     "modules" (condp = (py/python-type modules)
                                 :jvm-map-as-python (nn/ModuleDict modules)
                                 :jvm-iterable-as-python (nn/ModuleList modules)
                                 nil),
                     "params" (condp = (py/python-type parameters)
                                :jvm-iterable-as-python (nn/ParameterList
                                                         parameters)
                                :jvm-map-as-python (nn/ParameterDict parameters)
                                nil)})
                   nil))),
    "forward" (py-class/make-tuple-instance-fn
               (fn [self & args]
                 (let [fwd (py/get-attr self "fwd")] (apply cfn fwd args))))}))

(defn into-module
  ([fwd] (into-module fwd nil nil))
  ([fwd modules] (into-module fwd modules nil))
  ([fwd modules params] (-Module fwd modules params)))

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

(defn clone
  [n module-fn & args]
  (apply nn/Sequential (repeatedly n #(apply module-fn args))))

(defn chain
  [& modules]
  (->> modules
       (map as-module)
       flatten
       (apply cfn nn/Sequential)))

(defn concatenate
  [& modules]
  (let [bcast (apply juxt modules)
        fwd (fn [X]
              (-> X
                  bcast
                  torch/hstack))]
    (into-module fwd modules)))

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
(comment (maybe->parameters (nn/Linear 10 10))
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
        optim (resolve-py "torch.optim" (optim-name algorithm))
        kw (dissoc kw :algorithm)]
    (apply optim params (mapcat identity kw))))

;; Loss function
(defn as-loss-fn
  [lfn & {:as kw}]
  (let [loss-fn (resolve-py "torch.nn.functional"
                            (csk/->snake_case_string (name lfn)))]
    (fn [yhat y] (apply loss-fn yhat y (mapcat identity kw)))))

;; Training

(require-python '[torch.utils.data :refer [DataLoader]])
(require-python '[builtins :as python])

(defn- minibatch
  [model loss-fn optimizer device X Y]
  (with-gil-stack-rc-context (py. optimizer zero_grad)
                             (let [X (py. X to device)
                                   Y (py. Y to device)
                                   Yhat (model X)
                                   loss (loss-fn Yhat Y)]
                               (py. loss backward)
                               (py. optimizer step)
                               (py. loss item))))

(defn train
  ([dataset model loss-fn] (train dataset model loss-fn {}))
  ([dataset model loss-fn
    {:keys [optimizer loader parameters epochs device],
     :or {optimizer {:algorithm :adam-w}, loader {}, epochs 1}}]
   (let [dataset (maybe->tensor-dataset dataset)
         loader (apply DataLoader dataset (mapcat identity loader))
         parameters (or parameters [(optim-group model)])
         optimizer (as-optim parameters optimizer)
         device (or device "cpu")]
     (py. model train)
     (loop [epochs (range epochs)
            batches (seq (python/enumerate loader))
            history []]
       (if (seq batches)
         (let [[[i b] & batches] batches
               loss (apply minibatch model loss-fn optimizer device b)]
           (recur epochs
                  batches
                  (conj history {:epoch (first epochs), :batch i, :loss loss})))
         (if (seq epochs)
           (recur (rest epochs)
                  (-> loader
                      python/enumerate
                      seq)
                  history)
           {:model model, :history history}))))))


;!zprint {:format :skip}
(comment
  (require-python '[torch.nn.functional :as nnf])
  (let [p 10
        N 1000
        truth (nn/Linear p 1)
        X (torch/randn N p)
        y (py/with [_ (torch/no_grad)] (truth X))
        y (torch/add y (torch/mul 0.1 (torch/randn_like y)))
        ds (TensorDataset X y)
        trained (train ds (nn/Linear p 1)
                       (as-loss-fn :mse-loss :reduction "sum")
                       {:epochs 2
                        :optimizer {:algorithm :adam-w :lr 0.01}
                        :loader {:batch_size 32}
                        })]
    (clojure.pprint/pprint trained)
    (py/with [_ (torch/no_grad)]
             (-> (torch/sub
                  (py.- truth weight)
                  (py.- (:model trained) weight))
                 torch/abs
                 torch/sum)))

  (libpython-clj2.python.fn/args->pos-kw-args [[(py/->py-dict {"a" 1})] :lr 0.1]))
