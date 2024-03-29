#ifndef VTK_WRITER_HH
#define VTK_WRITER_HH VTK_WRITER_HH
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include "lattice/lattice.hh"

/** @file vtk_writer.hh
 *
 * @brief classes for writing fields to a vtk files
 */

/** @class VTKWriter
 *
 * @brief base class for writing fields to disk
 */
class VTKWriter
{
public:
    /** @brief Create a new instance
     *
     * @param[in] filename_ name of file to write to
     * @param[in] verbose_ verbosity level
     */
    VTKWriter(const std::string filename_,
              const int verbose_ = 0) : filename(filename_),
                                        verbose(verbose_) {}

    /** @brief Add state to collection of sample states to be written
     *
     * @param[in] phi state to write to disk
     * @param[in] label label to identify state in file
     */
    void add_state(const Eigen::VectorXd &phi, const std::string label);

    /** @brief write all sample states to disk */
    virtual void write() const = 0;

protected:
    /** @brief name of file to write */
    const std::string filename;
    /** @brief verbosity level */
    const int verbose;
    /** @brief dictionary of sample state to be written to disk
     * each state is identified by its label. */
    std::map<std::string, Eigen::VectorXd> sample_states;
};

#endif // VTK_WRITER_HH