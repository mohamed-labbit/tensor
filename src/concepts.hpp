#pragma once

#include <type_traits>


namespace internal {
namespace concepts {

template<class, class = std::void_t<>>
struct has_left_shift_operator: std::false_type
{
};

template<class _Tp>
struct has_left_shift_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_left_shift_operator_v = has_left_shift_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_right_shift_operator: std::false_type
{
};

template<class _Tp>
struct has_right_shift_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_right_shift_operator_v = has_right_shift_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_plus_operator: std::false_type
{
};

template<class _Tp>
struct has_plus_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_plus_operator_v = has_plus_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_minus_operator: std::false_type
{
};

template<class _Tp>
struct has_minus_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_minus_operator_v = has_minus_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_times_operator: std::false_type
{
};

template<class _Tp>
struct has_times_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_times_operator_v = has_times_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_divide_operator: std::false_type
{
};

template<class _Tp>
struct has_divide_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_divide_operator_v = has_divide_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() == std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_equal_operator_v = has_equal_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_not_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_not_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() != std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_not_equal_operator_v = has_not_equal_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_less_operator: std::false_type
{
};

template<typename _Tp>
struct has_less_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() < std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_less_operator_v = has_less_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_less_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_less_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() <= std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_less_equal_operator_v = has_less_equal_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_greater_operator: std::false_type
{
};

template<typename _Tp>
struct has_greater_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() > std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_greater_operator_v = has_greater_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_greater_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_greater_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() >= std::declval<_Tp>())>>
    : std::true_type
{
};

template<typename _Tp>
constexpr bool has_greater_equal_operator_v = has_greater_equal_operator<_Tp>::value;

}  // namespace concepts
}  // namespace internal
