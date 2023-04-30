(ns ptonner.pytorch-clj-test
  (:require [clojure.test :refer :all]
            [ptonner.init]
            [libpython-clj2.python :as py :refer
             [py.- py. with-gil-stack-rc-context path->py-obj]]
            [libpython-clj2.require :refer [require-python]]
            [ptonner.pytorch-clj :refer :all]))

(require-python '[builtins :as python])
(require-python '[torch])
(require-python '[torch.nn :as nn])

;; Modules


(defn- module-params
  [m]
  (-> m
      (py. parameters)
      py/->py-list))

(defn- params-match?
  [m1 m2]
  (reduce #(and %1 %2) (map = (module-params m1) (module-params m2))))

(deftest into-module-test
  (testing "Single module usage"
           (with-gil-stack-rc-context (let [m (nn/Linear 3 4)
                                            x (torch/randn 2 3)
                                            fwd (fn [x] (m x))
                                            mm (into-module fwd [m])]
                                        (is (torch/allclose (m x) (mm x)))
                                        (is (params-match? m mm))))))


;; Layers

(deftest add-test
  (testing
   "basic `add` layer"
   (with-gil-stack-rc-context
    (let [l1 (nn/Linear 3 4)
          l2 (nn/Linear 3 4)
          l3 (nn/Linear 3 4)
          x (torch/randn 2 3)
          a (add l1 l2 l3)
          out (a x)]
      (is (= (py.- out shape) (torch/Size [2 4])))
      (is (torch/allclose out (reduce torch/add [(l1 x) (l2 x) (l3 x)])))
      (is (params-match? a (nn/ModuleList [l1 l2 l3]))))))
  (testing
   "`add` layer with sequence input"
   (with-gil-stack-rc-context
    (let [mods (repeatedly 10 #(nn/Linear 3 4))
          x (torch/randn 2 3)
          a (add mods)]
      (is (torch/allclose (a x) (reduce torch/add (map #(% x) mods))))))))

(deftest clone-test
  (testing "basic `clone` usage"
           (let [c (clone 3 nn/Linear 3 3)
                 mods (python/list c)
                 x (torch/randn 3 3)]
             (is (torch/allclose (c x) (reduce #(%2 %1) x mods)))
             (is (= 6 (count (module-params c)))))))

(deftest chain-test
  (testing "basic `chain` usage"
           (let [c1 (nn/Linear 3 4)
                 c2 (nn/Linear 4 2)
                 c (chain c1 c2)
                 x (torch/randn 2 3)]
             (is (torch/allclose (c x)
                                 (-> x
                                     c1
                                     c2)))
             (is (params-match? c (nn/ModuleList [c1 c2])))))
  (testing "intermediate `chain` usage"
           (let [c1 (nn/Linear 3 4)
                 c2 (add (repeatedly 3 #(nn/Linear 4 2)))
                 c3 (clone 3 nn/Linear 2 2)
                 c (chain c1 c2 c3)
                 x (torch/randn 2 3)]
             (is (torch/allclose (c x)
                                 (-> x
                                     c1
                                     c2
                                     c3)))
             (is (= 14 (count (module-params c)))))))

(deftest concatenate-test
  (testing "basic `concatenate` usage"
           (is (with-gil-stack-rc-context
                (let [m1 (nn/Linear 2 3)
                      m2 (nn/Linear 2 4)
                      cct (concatenate m1 m2)
                      x (torch/randn 2 2)
                      out (cct x)]
                  (py/->jvm (= (py.- out shape) (torch/Size [2 7]))))))
           (is (with-gil-stack-rc-context
                (let [m1 (nn/Linear 2 3)
                      m2 (nn/Linear 2 4)
                      cct (concatenate m1 m2)
                      x (torch/randn 2 2)
                      out (cct x)]
                  (py/->jvm (params-match? cct (nn/ModuleList [m1 m2]))))))))


(deftest residual-test
  (testing "Basic `residual` usage"
           (is (with-gil-stack-rc-context
                (let [m (nn/Linear 3 3)
                      x (torch/randn 3 3)
                      r (residual m)
                      m-out (m x)
                      r-out (r x)]
                  (py/->jvm (torch/allclose (torch/sub r-out x) m-out)))))
           (is (with-gil-stack-rc-context (let [m (nn/Linear 3 3)
                                                r (residual m)]
                                            (py/->jvm (params-match? m r)))))))

;; Optim

(deftest maybe->parameters-test
  (testing
   "Module parameters returned"
   (is (with-gil-stack-rc-context (let [l (nn/Linear 10 10)
                                        weight (-> l
                                                   maybe->parameters
                                                   py/->py-list
                                                   first)]
                                    (py/->jvm (= weight) (py.- l weight)))))))

(deftest as-optim-test
  (testing
   "Correct algorithm returned"
   (are [alg opt nm]
        (py/is-instance? (as-optim [(nn/Parameter)] (assoc opt :algorithm alg))
                         (path->py-obj (str "torch.optim." nm)))
        :adam
        {}
        "Adam"
        :adam-w
        {}
        "AdamW"
        :rmsprop
        {}
        "RMSprop"
        :sgd
        {:lr 0.1}
        "SGD"
        :asgd
        {:lr 0.1}
        "ASGD"
        :lbfgs
        {}
        "LBFGS")))
