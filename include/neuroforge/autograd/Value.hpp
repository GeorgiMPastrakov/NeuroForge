#pragma once

#include <memory>

namespace neuroforge {

class Value {
public:
    struct Node;

    Value(double data, bool requires_grad = false);
    explicit Value(std::shared_ptr<Node> node);

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
    std::shared_ptr<Node> node_;
};

}
