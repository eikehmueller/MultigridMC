#ifndef LATTICE2D_HH
#define LATTICE2D_HH LATTICE2D_HH
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "lattice.hh"

/** @file lattice2d.hh
 *
 * @brief two dimensional lattice
 */

/** @class Lattice2d
 *
 * @brief Two dimensional structured lattice of size nx x ny with periodic boundary
 *
 * Periodic boundary conditions are implicitly assumed
 *
 *  Points are arranged ordered lexicographically, for example for nx = 4, ny =3:
 *
 *  ^ y
 *  !
 *                                         N
 *  8 ---- 9 --- 10 --- 11              W  +  E
 *  !      !      !      !                 S
 *  !      !      !      !
 *  4 ---- 5 ---- 6 ---- 7
 *  !      !      !      !
 *  !      !      !      !
 *  0 ---- 1 ---- 2 ---- 3  ---> x
 *
 */
class Lattice2d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] nx_ Extent in x-direction
   * @param[in] ny_ Extent in y-direction
   */
  Lattice2d(const unsigned int nx_, const unsigned int ny_) : nx(nx_), ny(ny_), Lattice(nx_ * ny_) {}

  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi idx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < M);
    Eigen::VectorXi idx(2);
    idx[0] = ell % nx;
    idx[1] = ell / nx;
    return idx;
  }

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int idx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    return ((idx[1] + ny) % ny) * nx + ((idx[0] + nx) % nx);
  };

  /** @brief Shift a linear index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_index(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = (ell % nx) + shift[0];
    int j = (ell / nx) + shift[1];
    return ((j + ny) % ny) * nx + ((i + nx) % nx);
  };

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    assert(nx % 2 == 0);
    assert(ny % 2 == 0);
    return std::make_shared<Lattice2d>(nx / 2, ny / 2);
  };

  /** @brief get info string */
  virtual std::string get_info() const;

  /** @brief extent in x-direction */
  const unsigned int nx;
  /** @brief extent in y-direction */
  const unsigned int ny;
};

#endif // LATTICE2D_HH