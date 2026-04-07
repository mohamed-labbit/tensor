template<class _Tp>
class tensor: public TensorBase<_Tp, std::vector<_Tp>>
{
 public:
  arch::tensor() = default;
  arch::tensor(const arch::tensor<_Tp>& t);
  arch::tensor(tensor&& t) TENSOR_NOEXCEPT;
  arch::tensor(const shape::Shape& shape_, const tensor& other);
  arch::tensor(const shape::Shape&                     shape_,
               std::initializer_list<Base::value_type> init_list,
               std::optional<Device>                   d = Device::CPU);

  explicit arch::tensor(const shape::Shape& shape_, const Base::value_type& v, std::optional<Device> d = Device::CPU);
  explicit arch::tensor(const shape::Shape& shape_, std::optional<Device> d = Device::CPU);
  explicit arch::tensor(const shape::Shape&                  shape_,
                        const typename Base::container_type& d,
                        std::optional<Device>                dev = Device::CPU);

  arch::tensor<_s16> int16_() const;
  arch::tensor<_s32> int32_() const;
  arch::tensor<_s64> int64_() const;
  arch::tensor<_u32> uint32_() const;
  arch::tensor<_u64> uint64_() const;
  arch::tensor<_f32> float32_() const;
  arch::tensor<_f64> float64_() const;
  unsigned long long count_nonzero(std::optional<unsigned long long> dimension = 0) const;
  unsigned long long lcm() const;
  arch::tensor<_Tp>  lcm(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> bool_() const;
  TENSOR_LIBRARY_API arch::tensor<bool> logical_not() const;
  TENSOR_LIBRARY_API arch::tensor<bool> logical_or(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<bool> logical_or(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor logical_xor(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor logical_xor(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor logical_and(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor logical_and(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_not_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_or_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_or_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_xor_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_xor_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_and_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& logical_and_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<bool> less_equal(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> less_equal(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<bool> greater_equal(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> greater_equal(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<bool> equal(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> equal(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<bool> not_equal(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> not_equal(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<bool> less(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> less(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor<bool> greater(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<bool> greater(const Base::value_type value) const;
  TENSOR_LIBRARY_API bool               operator==(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API bool               operator!=(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor operator+(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor operator-(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor operator+(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor operator-(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor operator*(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor operator*(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator-=(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator+=(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator*=(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator/=(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator+=(const Base::value_type& value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator-=(const Base::value_type& value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator/=(const Base::value_type& value);
  TENSOR_LIBRARY_API arch::tensor operator/(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor operator/(const Base::value_type& value) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator*=(const Base::value_type& value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator=(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator=(arch::tensor<_Tp>&& other) const TENSOR_NOEXCEPT;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& operator=(const arch::tensor<_Tp>&) = default;
  TENSOR_LIBRARY_API arch::tensor<bool>& operator!();
  TENSOR_LIBRARY_API const arch::tensor<bool>& operator!() const;
  TENSOR_LIBRARY_API arch::tensor slice(unsigned long long                dimension,
                                        std::optional<unsigned long long> start,
                                        std::optional<unsigned long long> end,
                                        int64_t                           step) const;
  TENSOR_LIBRARY_API arch::tensor fmax(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor fmax(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor fmod(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor fmod(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor frac() const;
  TENSOR_LIBRARY_API arch::tensor log10() const;
  TENSOR_LIBRARY_API arch::tensor log2() const;
  TENSOR_LIBRARY_API arch::tensor exp() const;
  TENSOR_LIBRARY_API arch::tensor sqrt() const;
  TENSOR_LIBRARY_API arch::tensor row(const unsigned long long index) const;
  TENSOR_LIBRARY_API arch::tensor col(const unsigned long long index) const;
  TENSOR_LIBRARY_API arch::tensor ceil() const;
  TENSOR_LIBRARY_API arch::tensor floor() const;
  TENSOR_LIBRARY_API arch::tensor clone() const;
  TENSOR_LIBRARY_API arch::tensor clamp(
    std::optional<const Base::value_type&> min_val = std::numeric_limits<Base::value_type>::lowest(),
    std::optional<const Base::value_type&> max_val = std::numeric_limits<Base::value_type>::max()) const;
  TENSOR_LIBRARY_API arch::tensor cos() const;
  TENSOR_LIBRARY_API arch::tensor cosh() const;
  TENSOR_LIBRARY_API arch::tensor acos() const;
  TENSOR_LIBRARY_API arch::tensor acosh() const;
  TENSOR_LIBRARY_API arch::tensor tan() const;
  TENSOR_LIBRARY_API arch::tensor tanh() const;
  TENSOR_LIBRARY_API arch::tensor atan() const;
  TENSOR_LIBRARY_API arch::tensor atanh() const;
  TENSOR_LIBRARY_API arch::tensor sin() const;
  TENSOR_LIBRARY_API arch::tensor sinc() const;
  TENSOR_LIBRARY_API arch::tensor sinh() const;
  TENSOR_LIBRARY_API arch::tensor asin() const;
  TENSOR_LIBRARY_API arch::tensor asinh() const;
  TENSOR_LIBRARY_API arch::tensor abs() const;
  TENSOR_LIBRARY_API arch::tensor bitwise_not() const;
  TENSOR_LIBRARY_API arch::tensor bitwise_and(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_and(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_or(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_or(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_xor(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_xor(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_left_shift(const int amount) const;
  TENSOR_LIBRARY_API arch::tensor bitwise_right_shift(const int amount) const;
  TENSOR_LIBRARY_API arch::tensor matmul(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor reshape(const shape::Shape shape) const;
  TENSOR_LIBRARY_API arch::tensor reshape_as(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor cross_product(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor absolute(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor dot(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor relu() const;
  TENSOR_LIBRARY_API arch::tensor fill(const arch::Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor fill(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor resize_as(const shape::Shape shape_) const;
  TENSOR_LIBRARY_API arch::tensor all() const;
  TENSOR_LIBRARY_API arch::tensor any() const;
  TENSOR_LIBRARY_API arch::tensor det() const;
  TENSOR_LIBRARY_API arch::tensor square() const;
  TENSOR_LIBRARY_API arch::tensor sigmoid() const;
  TENSOR_LIBRARY_API arch::tensor clipped_relu(const Base::value_type clip_limit) const;
  TENSOR_LIBRARY_API arch::tensor remainder(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor remainder(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor maximum(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor maximum(const Base::value_type& value) const;
  TENSOR_LIBRARY_API arch::tensor dist(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor dist(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor negative() const;
  TENSOR_LIBRARY_API arch::tensor permute(const unsigned long long dimension) const;
  TENSOR_LIBRARY_API arch::tensor gcd(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API arch::tensor gcd(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor pow(const arch::tensor<_Tp>& other) const;
  TENSOR_LIBRARY_API tensor       pow(const Base::value_type value) const;
  TENSOR_LIBRARY_API arch::tensor cumprod(unsigned long long dimension = -1) const;
  TENSOR_LIBRARY_API arch::tensor cat(const std::vector<arch::tensor>& _others, unsigned long long _dim) const;
  TENSOR_LIBRARY_API arch::tensor argmax(unsigned long long dimension) const;
  TENSOR_LIBRARY_API arch::tensor unsqueeze(unsigned long long dimension) const;
  TENSOR_LIBRARY_API arch::tensor zeros(const shape::Shape& shape_);
  TENSOR_LIBRARY_API arch::tensor ones(const shape::Shape& shape_);
  TENSOR_LIBRARY_API arch::tensor randomize(const shape::Shape& shape_, bool bounded = false);
  TENSOR_LIBRARY_API arch::tensor get_minor(unsigned long long a, unsigned long long b) const;
  TENSOR_LIBRARY_API arch::tensor expand_as(shape::Shape shape_, unsigned long long dimension) const;
  TENSOR_LIBRARY_API arch::tensor lcm(const shape::Shape& other) const;
  TENSOR_LIBRARY_API double       mean() const;
  TENSOR_LIBRARY_API double       median(const unsigned long long dimension) const;
  TENSOR_LIBRARY_API double       mode(const unsigned long long dimension) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& push_back(Base::value_type v) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& pop_back() const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& sqrt_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& exp_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& log2_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& log10_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& frac_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& fmod_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& fmod_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& cos_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& cosh_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& acos_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& acosh_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& tan_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& tanh_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& atan_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& atanh_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& sin_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& sinh_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& asin_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& asinh_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& ceil_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& floor_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& relu_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& clamp_(
    std::optional<const Base::value_type&> min_val = std::numeric_limits<Base::value_type>::lowest(),
    std::optional<const Base::value_type&> max_val = std::numeric_limits<Base::value_type>::max());
  TENSOR_LIBRARY_API arch::tensor clamp_min(const Base::value_type& min_val) const;
  TENSOR_LIBRARY_API arch::tensor clamp_max(const Base::value_type& max_val) const;
  TENSOR_LIBRARY_API arch::tensor<_Tp>& clamp_min_(const Base::value_type& min_val);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& clamp_max_(const Base::value_type& max_val);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& abs_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& permute_(const unsigned long long dimension);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& repeat_(const typename Base::container_type& d,
                                                std::optional<int>                   dimension = 0);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& negative_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& transpose_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& unsqueeze_(unsigned long long dimension);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& squeeze_(unsigned long long dimension);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& resize_as_(const shape::Shape sh_);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& dist_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& dist_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& maximum_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& maximum_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& remainder_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& remainder_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& fill_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& fill_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& sigmoid_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& clipped_relu_(const Base::value_type clip_limit);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& square_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& pow_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& pow_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& sinc_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_left_shift_(const int amount);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_right_shift_(const int amount);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_and_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_and_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_or_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_or_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_xor_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_xor_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& bitwise_not_();
  TENSOR_LIBRARY_API arch::tensor<_Tp>& view(std::initializer_list<unsigned long long> new_shape);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& fmax_(const arch::tensor<_Tp>& other);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& fmax_(const Base::value_type value);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& randomize_(const shape::Shape& sh, std::optional<bool> bounded = false);
  TENSOR_LIBRARY_API arch::tensor<_Tp>& zeros_(std::optional<shape::Shape> sh = {});
  TENSOR_LIBRARY_API arch::tensor<_Tp>& ones_(std::optional<shape::Shape> shape_ = {});
  TENSOR_LIBRARY_API arch::tensor<unsigned long long> argmax_(unsigned long long dimension) const;
  TENSOR_LIBRARY_API arch::tensor<unsigned long long> argsort(std::optional<unsigned long long> dimension = -1,
                                                              std::optional<bool>               ascending = true) const;

 private:
  TENSOR_NODISCARD std::size_t computeStride(std::size_t                            dimension,
                                             const std::vector<unsigned long long>& shape) const TENSOR_NOEXCEPT;
  void printRecursive(std::size_t index, std::size_t depth, const std::vector<unsigned long long>& shape) const;
  TENSOR_NODISCARD unsigned long long        compute_index(const std::vector<unsigned long long>& idx) const;
  TENSOR_NODISCARD static unsigned long long computeSize(const std::vector<unsigned long long>& dims) TENSOR_NOEXCEPT;
  unsigned long long                         compute_outer_size(const unsigned long long dimension) const;
  TENSOR_NODISCARD static _f32               frac(const Base::value_type& value) TENSOR_NOEXCEPT;
  // where the tensor is stored
  bool is_cuda_device() const;
};  // tensor class
