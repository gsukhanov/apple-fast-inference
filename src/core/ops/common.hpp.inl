namespace fastinf {
template <typename TLhs, typename TRhs>
void check_same_shape(const TLhs& lhs, const TRhs& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error(
            "The size of tensor a must match the size of tensor b at "
            "non-singleton dimension");
    }
}
}  // namespace fastinf
