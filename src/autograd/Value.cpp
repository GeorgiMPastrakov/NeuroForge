#include "neuroforge/autograd/Value.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace neuroforge {

struct Value::Node {
    double data = 0.0;
    double grad = 0.0;
    double backward_grad = 0.0;
    bool requires_grad = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward;
};

namespace {

void buildTopo(const std::shared_ptr<Value::Node>& node, std::unordered_set<Value::Node*>& visited, std::vector<std::shared_ptr<Value::Node>>& order) {
    if (visited.count(node.get()) > 0) {
        return;
    }

    visited.insert(node.get());

    for (const auto& parent : node->parents) {
        buildTopo(parent, visited, order);
    }

    order.push_back(node);
}

Value makeValue(double data, bool requires_grad, std::vector<std::shared_ptr<Value::Node>> parents, std::function<void(const std::shared_ptr<Value::Node>&)> backward) {
    auto node = std::make_shared<Value::Node>();
    node->data = data;
    node->requires_grad = requires_grad;
    node->parents = std::move(parents);
    std::weak_ptr<Value::Node> weak = node;
    node->backward = [weak, backward] {
        if (auto current = weak.lock()) {
            backward(current);
        }
    };
    return Value(node);
}

}

Value::Value(double data, bool requires_grad)
    : node_(std::make_shared<Node>()) {
    node_->data = data;
    node_->requires_grad = requires_grad;
}

Value::Value(std::shared_ptr<Node> node)
    : node_(std::move(node)) {
}

double Value::data() const {
    return node_->data;
}

double Value::grad() const {
    return node_->grad;
}

bool Value::requiresGrad() const {
    return node_->requires_grad;
}

void Value::zeroGrad() {
    node_->grad = 0.0;
    node_->backward_grad = 0.0;
}

void Value::backward() {
    if (!requiresGrad()) {
        return;
    }

    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    buildTopo(node_, visited, order);

    for (const auto& node : order) {
        node->backward_grad = 0.0;
    }

    node_->backward_grad = 1.0;

    for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
        if ((*iterator)->backward) {
            (*iterator)->backward();
        }
    }

    for (const auto& node : order) {
        if (node->requires_grad) {
            node->grad += node->backward_grad;
        }
    }
}

Value Value::pow(double exponent) const {
    const bool requires_grad = requiresGrad();
    auto parent = node_;
    return makeValue(std::pow(data(), exponent), requires_grad, {parent}, [parent, exponent](const std::shared_ptr<Node>& current) {
        if (parent->requires_grad) {
            parent->backward_grad += current->backward_grad * exponent * std::pow(parent->data, exponent - 1.0);
        }
    });
}

Value Value::relu() const {
    const bool requires_grad = requiresGrad();
    auto parent = node_;
    return makeValue(data() > 0.0 ? data() : 0.0, requires_grad, {parent}, [parent](const std::shared_ptr<Node>& current) {
        if (parent->requires_grad) {
            parent->backward_grad += parent->data > 0.0 ? current->backward_grad : 0.0;
        }
    });
}

Value Value::sigmoid() const {
    const double output = 1.0 / (1.0 + std::exp(-data()));
    const bool requires_grad = requiresGrad();
    auto parent = node_;
    return makeValue(output, requires_grad, {parent}, [parent, output](const std::shared_ptr<Node>& current) {
        if (parent->requires_grad) {
            parent->backward_grad += current->backward_grad * output * (1.0 - output);
        }
    });
}

Value Value::tanh() const {
    const double output = std::tanh(data());
    const bool requires_grad = requiresGrad();
    auto parent = node_;
    return makeValue(output, requires_grad, {parent}, [parent, output](const std::shared_ptr<Node>& current) {
        if (parent->requires_grad) {
            parent->backward_grad += current->backward_grad * (1.0 - output * output);
        }
    });
}

Value operator+(const Value& left, const Value& right) {
    const bool requires_grad = left.requiresGrad() || right.requiresGrad();
    auto left_node = left.node_;
    auto right_node = right.node_;
    return makeValue(left.data() + right.data(), requires_grad, {left_node, right_node}, [left_node, right_node](const std::shared_ptr<Value::Node>& current) {
        if (left_node->requires_grad) {
            left_node->backward_grad += current->backward_grad;
        }

        if (right_node->requires_grad) {
            right_node->backward_grad += current->backward_grad;
        }
    });
}

Value operator-(const Value& left, const Value& right) {
    const bool requires_grad = left.requiresGrad() || right.requiresGrad();
    auto left_node = left.node_;
    auto right_node = right.node_;
    return makeValue(left.data() - right.data(), requires_grad, {left_node, right_node}, [left_node, right_node](const std::shared_ptr<Value::Node>& current) {
        if (left_node->requires_grad) {
            left_node->backward_grad += current->backward_grad;
        }

        if (right_node->requires_grad) {
            right_node->backward_grad -= current->backward_grad;
        }
    });
}

Value operator*(const Value& left, const Value& right) {
    const bool requires_grad = left.requiresGrad() || right.requiresGrad();
    auto left_node = left.node_;
    auto right_node = right.node_;
    return makeValue(left.data() * right.data(), requires_grad, {left_node, right_node}, [left_node, right_node](const std::shared_ptr<Value::Node>& current) {
        if (left_node->requires_grad) {
            left_node->backward_grad += current->backward_grad * right_node->data;
        }

        if (right_node->requires_grad) {
            right_node->backward_grad += current->backward_grad * left_node->data;
        }
    });
}

Value operator/(const Value& left, const Value& right) {
    if (right.data() == 0.0) {
        throw std::invalid_argument("Value division by zero.");
    }

    const bool requires_grad = left.requiresGrad() || right.requiresGrad();
    auto left_node = left.node_;
    auto right_node = right.node_;
    return makeValue(left.data() / right.data(), requires_grad, {left_node, right_node}, [left_node, right_node](const std::shared_ptr<Value::Node>& current) {
        if (left_node->requires_grad) {
            left_node->backward_grad += current->backward_grad / right_node->data;
        }

        if (right_node->requires_grad) {
            right_node->backward_grad -= current->backward_grad * left_node->data / (right_node->data * right_node->data);
        }
    });
}

}
