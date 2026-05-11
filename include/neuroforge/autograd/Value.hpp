#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace neuroforge {

class Value {
public:
    Value(double data, bool requires_grad = false);

    double data() const;
    double grad() const;
    bool requiresGrad() const;
    void zeroGrad();
    void backward();

    Value pow(double exponent) const;
    Value relu() const;
    Value sigmoid() const;
    Value tanh() const;

    friend Value operator+(const Value& left, const Value& right);
    friend Value operator-(const Value& left, const Value& right);
    friend Value operator*(const Value& left, const Value& right);
    friend Value operator/(const Value& left, const Value& right);
private:
    struct Node;

    explicit Value(std::shared_ptr<Node> node);

    static Value make(double data, bool requires_grad, std::vector<std::shared_ptr<Node>> parents, std::function<void(const std::shared_ptr<Node>&)> backward);

    std::shared_ptr<Node> node_;
};

}
