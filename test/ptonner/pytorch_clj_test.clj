(ns ptonner.pytorch-clj-test
  (:require [clojure.test :refer :all]
            [ptonner.init]
            [libpython-clj2.python :as py :refer
             [py.- with-gil-stack-rc-context]]
            [libpython-clj2.require :refer [require-python]]
            [ptonner.pytorch-clj :refer :all]))

(require-python '[builtins :as python])
(require-python '[torch])
(require-python '[torch.nn :as nn])

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
      (is (torch/allclose out (reduce torch/add [(l1 x) (l2 x) (l3 x)]))))))
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
             (is (torch/allclose (c x) (reduce #(%2 %1) x mods))))))

(deftest chain-test
  (testing "basic `chain` usage"
           (let [c1 (nn/Linear 3 4)
                 c2 (add (repeatedly 3 #(nn/Linear 4 2)))
                 c (chain c1 c2)
                 x (torch/randn 2 3)]
             (is (torch/allclose (c x)
                                 (-> x
                                     c1
                                     c2)))))
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
                                     c3))))))

(deftest concatenate-test
  (testing "basic `concatenate` usage"
           (is (with-gil-stack-rc-context
                (let [m1 (nn/Linear 2 3)
                      m2 (nn/Linear 2 4)
                      cct (concatenate m1 m2)
                      x (torch/randn 2 2)
                      out (cct x)]
                  (py/->jvm (= (py.- out shape) (torch/Size [2 7]))))))))
