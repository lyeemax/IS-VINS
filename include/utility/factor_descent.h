#ifndef FACTOR_DESCENT_H_
#define FACTOR_DESCENT_H_

//wolf includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <Eigen/Sparse>
using namespace Eigen;
typedef Eigen::MatrixXd MatrixXs;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorXs;
struct PruningOptimOptions
{
    unsigned int max_iter_;
    int max_time_;
    int min_gradient_;

    // FD options
    int cyclic_;
    bool apply_closed_form_;
    int formulation_;
 };

struct ConstraintBasePtr{
    int getSize(){

    }
};
Eigen::MatrixXi rowsAndCols(const std::map<ConstraintBasePtr,unsigned int>& ctr_2_row, const unsigned int& size, std::map<ConstraintBasePtr,unsigned int>& ctr_2_ind)
{
    Eigen::MatrixXi rows_and_cols(0,2);
    unsigned int j = 0;
    unsigned int dim, block_size;
    for (auto pair : ctr_2_row)
    {
        dim = pair.first->getSize();
        block_size = (dim+1)*dim / 2;

        rows_and_cols.conservativeResize(rows_and_cols.rows()+block_size,2);
        ctr_2_ind[pair.first] = j;

        for (auto row = 0; row < dim; row++)
            for (auto col = row; col < dim; col++)
            {
                rows_and_cols(j,0) = pair.second + row;
                rows_and_cols(j,1) = pair.second + col;
                j++;
            }

    }
    return rows_and_cols;
}

void Matrix2vector(const MatrixXs& _Omega, VectorXs& x_, const Eigen::MatrixXi& _rows_and_cols)
{
    for (auto i = 0; i < _rows_and_cols.rows(); i++)
        x_(i) = _Omega(_rows_and_cols(i,0), _rows_and_cols(i,1));
}

template<int _Options, typename _StorageIndex>
void assignSparseBlock(const MatrixXs& ins, Eigen::SparseMatrix<int,_Options,_StorageIndex>& original, const unsigned int& row, const unsigned int& col)
{
    assert(original.rows() >= row + ins.rows() && "wrong sparse matrices sizes");
    assert(original.cols() >= col + ins.cols() && "wrong sparse matrices sizes");

    for (auto ins_row = 0; ins_row < ins.rows(); ins_row++)
        for (auto ins_col = 0; ins_col < ins.cols(); ins_col++)
            original.coeffRef(row+ins_row, col+ins_col) = ins(ins_row,ins_col);

    original.makeCompressed();
}

void insertSparseBlock(const MatrixXs& ins, MatrixXs& original, const unsigned int& row, const unsigned int& col)
{
    assert(original.rows() >= row + ins.rows() && "wrong sparse matrices sizes");
    assert(original.cols() >= col + ins.cols() && "wrong sparse matrices sizes");

    for (auto ins_row = 0; ins_row < ins.rows(); ins_row++)
        for (auto ins_col = 0; ins_col < ins.cols(); ins_col++)
            original.insert(row+ins_row, col+ins_col) = ins(ins_row,ins_col);

    original.makeCompressed();
}

void updateUpsilon(const std::map<ConstraintBasePtr,unsigned int>::const_iterator& ctr_pair_it, const MatrixXs& J, const MatrixXs& Omega, const unsigned int& omega_size, MatrixXs& Upsilon_k)
{
    Upsilon_k.setZero();
    if (ctr_pair_it->second != 0)
        Upsilon_k += J.topRows(ctr_pair_it->second).transpose() *
                     Omega.topLeftCorner(ctr_pair_it->second,ctr_pair_it->second) *
                     J.topRows(ctr_pair_it->second);

    unsigned int next_block_size = omega_size - ctr_pair_it->second - ctr_pair_it->first->getSize();
    if (next_block_size != 0)
        Upsilon_k += J.bottomRows(next_block_size).transpose() *
                     Omega.bottomRightCorner(next_block_size,next_block_size) *
                     J.bottomRows(next_block_size);
}

void getGradientDirection(VectorXs KLD_gradient, const std::map<ConstraintBasePtr,unsigned int>& ctr_2_row, std::map<ConstraintBasePtr,unsigned int>& ctr_2_ind, const std::map<ConstraintBasePtr,bool> closed_form_factor, const std::map<ConstraintBasePtr,bool> ctr_2_spd, std::map<ConstraintBasePtr,unsigned int>::const_iterator& ctr_pair_it)
{
    // non-cyclic max gradient norm
    auto current_pair = ctr_pair_it;

    // find max gradient norm factor
    // bigger than 1e-3, otherwise take the next factor
    int max_sq_norm = 1e-6;
    ctr_pair_it++;
    if (ctr_pair_it == ctr_2_row.end())
        ctr_pair_it = ctr_2_row.begin();
    for (auto pair_it = ctr_2_row.begin(); pair_it != ctr_2_row.end(); pair_it++)
    {
        // NOT CANDIDATES: current factor & closed form factors (& SDP factors)
        if (pair_it == current_pair ||
            (closed_form_factor.find(pair_it->first) != closed_form_factor.end() && closed_form_factor.at(pair_it->first)))
            continue;

        int dim = pair_it->first->getSize();
        int block_size = (dim+1)*dim / 2;
        int sq_norm = KLD_gradient.segment(ctr_2_ind[pair_it->first],block_size).squaredNorm();

        if (sq_norm > max_sq_norm)
        {
            max_sq_norm = sq_norm;
            ctr_pair_it = pair_it;
        }
    }
}

void computeOmegaK(const std::map<ConstraintBasePtr,unsigned int>::const_iterator& ctr_pair_it,
                   const std::map<ConstraintBasePtr,MatrixXs>& ctr_2_Phi,
                   std::map<ConstraintBasePtr,MatrixXs>& ctr_2_QL,
                   std::map<ConstraintBasePtr,MatrixXs>& ctr_2_Q0,
                   std::map<ConstraintBasePtr,MatrixXs>& ctr_2_iL,
                   MatrixXs& Ji,
                   Eigen::HouseholderQR<MatrixXs>& qr,
                   MatrixXs& Q,
                   MatrixXs& R,
                   Eigen::ColPivHouseholderQR<MatrixXs>& Upsilon_decomp,
                   const MatrixXs& Upsilon_k,
                   const MatrixXs& J,
                   const PruningOptimOptions& options,
                   MatrixXs& Omega_k,
                   MatrixXs& Omega_k2)
{
    bool computed = false;

    ////////////////////// FIRST TERM Omega_k constant part /////////////////////////////////////////////
    //std::cout << "Omega_k constant part\n";
    Omega_k = ctr_2_Phi.at(ctr_pair_it->first);

    ////////////////////// SECOND TERM Omega_k variable part ////////////////////////////////////////////
    //std::cout << "Omega_k variable part\n";

    Upsilon_decomp.compute(Upsilon_k);

    // PROPOSITION 2: Upsilon invertible ==================================
    if (Upsilon_decomp.isInvertible())
        switch (options.formulation_)
        {
            case 0:
            {
                Omega_k += -(J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()) *
                             Upsilon_decomp.inverse() *
                             J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).transpose()).inverse();
                computed = true;
                break;
            }
            case 1:
            {
                Omega_k += -(J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()) * 
                             Upsilon_decomp.solve(J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).transpose())).inverse();

                if (!Omega_k.allFinite())
                {
                    std::cout << "Factor Descent (Upsilon invertible)---------------- Any NaN or Inf in Omega_k!\n";
                    Eigen::SelfAdjointEigenSolver<MatrixXs> es(Upsilon_k);
                    std::cout << "Upsilon_k eigenvalues: " << es.eigenvalues().transpose() << std::endl;
                    std::cout << "NaN in Omega_k: " << Omega_k.hasNaN() << std::endl;
                    std::cout << "NaN in L: " << MatrixXs(Upsilon_k.llt().matrixL()).hasNaN() << std::endl;
                    std::cout << "NaN in J: " << J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).hasNaN() << std::endl;
                    //std::cout << "NaN in iLJt: " << iLJt.hasNaN() << std::endl;
                    //std::cout << "NaN in iLJt.transpose() * iLJt: " << (iLJt.transpose() * iLJt).hasNaN() << std::endl;
                    //std::cout << "NaN in (iLJt.transpose() * iLJt).inverse(): " << (iLJt.transpose() * iLJt).inverse().hasNaN() << std::endl;
                    std::cout << "Inf in Omega_k: " << !Omega_k.allFinite() << std::endl;
                    std::cout << "Inf in L: " << !MatrixXs(Upsilon_k.llt().matrixL()).allFinite() << std::endl;
                    std::cout << "Inf in J: " << !J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).allFinite() << std::endl;
                    //std::cout << "Inf in iLJt: " << !iLJt.allFinite() << std::endl;
                    //std::cout << "Inf in iLJt.transpose() * iLJt: " << !(iLJt.transpose() * iLJt).allFinite() << std::endl;
                    //std::cout << "Inf in (iLJt.transpose() * iLJt).inverse(): " << !(iLJt.transpose() * iLJt).inverse().allFinite() << std::endl;
                    std::cout << "Omega_k:\n" << Omega_k << std::endl;
                    //std::cout << "iLJt: \n" << iLJt << std::endl;
                    //std::cout << "iLJt.transpose() * iLJt: \n" << (iLJt.transpose() * iLJt) << std::endl;
                    //std::cout << "(iLJt.transpose() * iLJt).inverse(): \n" << (iLJt.transpose() * iLJt).inverse() << std::endl;
                }
                else
                    computed = true;

                break;
            }
            default:
                std::cout << "Unknown formulation!" << std::endl;
        }

    // PROPOSITION 1: Upsilon not invertible ===================================================
    if (!computed)
    {
        // initialize QL decomposition of J if not done before
        if (ctr_2_QL.find(ctr_pair_it->first) == ctr_2_QL.end())
        {
            Ji = J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize());
            qr.compute(Ji.transpose());
            Q = qr.householderQ().transpose();
            ctr_2_QL[ctr_pair_it->first] = Q.topRows(ctr_pair_it->first->getSize());
            ctr_2_Q0[ctr_pair_it->first] = Q.bottomRows(Q.rows() - ctr_pair_it->first->getSize());
            R = qr.matrixQR().triangularView<Eigen::Upper>();
            ctr_2_iL[ctr_pair_it->first] = R.topRows(ctr_pair_it->first->getSize()).transpose().inverse();
        }

        if ((ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).determinant() > 1e9) // Projection of Upsilon invertible (Lambda invertible actually)
        {
            Omega_k2 = -ctr_2_iL.at(ctr_pair_it->first).transpose() * ctr_2_QL.at(ctr_pair_it->first) * (Upsilon_k - Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose() * (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse() * ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k) * ctr_2_QL.at(ctr_pair_it->first).transpose() * ctr_2_iL.at(ctr_pair_it->first);

            if (!Omega_k2.allFinite())
            {
                std::cout << "Factor Descent (Upsilon not invertible)---------------- Any NaN or Inf in Omega_k!\n";
                std::cout << "NaN in Omega_k: " << Omega_k.hasNaN() << std::endl;
                std::cout << "NaN in Omega_k2: " << Omega_k2.hasNaN() << std::endl;
                std::cout << "NaN in iL: " << ctr_2_iL.at(ctr_pair_it->first).hasNaN() << std::endl;
                std::cout << "NaN in Upsilon_k: " << Upsilon_k.hasNaN() << std::endl;
                std::cout << "NaN in Q0: " << ctr_2_Q0.at(ctr_pair_it->first).hasNaN() << std::endl;
                std::cout << "NaN in aux: " << (Upsilon_k - Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose() * (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse() * ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k).hasNaN() << std::endl;
                std::cout << "NaN in aux2: " << (ctr_2_Q0.at(ctr_pair_it->first).transpose() * (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse() * ctr_2_Q0.at(ctr_pair_it->first)).hasNaN() << std::endl;
                std::cout << "NaN in aux3: " << (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse().hasNaN() << std::endl;
                std::cout << "NaN in aux4: " << (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).hasNaN() << std::endl;
                std::cout << "Inf in Omega_k: " << !Omega_k.allFinite() << std::endl;
                std::cout << "Inf in Omega_k2: " << !Omega_k2.allFinite() << std::endl;
                std::cout << "Inf in iL: " << !ctr_2_iL.at(ctr_pair_it->first).allFinite() << std::endl;
                std::cout << "Inf in Upsilon_k: " << !Upsilon_k.allFinite() << std::endl;
                std::cout << "Inf in Q0: " << !ctr_2_Q0.at(ctr_pair_it->first).allFinite() << std::endl;
                std::cout << "Inf in aux: " << !(Upsilon_k - Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose() * (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse() * ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k).allFinite() << std::endl;
                std::cout << "Inf in aux2: " << !(ctr_2_Q0.at(ctr_pair_it->first).transpose() * (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse() * ctr_2_Q0.at(ctr_pair_it->first)).allFinite() << std::endl;
                std::cout << "Inf in aux3: " << !(ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).inverse().allFinite() << std::endl;
                std::cout << "Inf in aux4: " << !(ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).allFinite() << std::endl;
                std::cout << "det(Q0 * Upsilon_k * Q0'): " << (ctr_2_Q0.at(ctr_pair_it->first)*Upsilon_k*ctr_2_Q0.at(ctr_pair_it->first).transpose()).determinant() << std::endl;
            }
            else
                Omega_k += Omega_k2;
        }
    }
}

void initPhiK(const std::map<ConstraintBasePtr,unsigned int>::const_iterator& ctr_pair_it,
              const MatrixXs& J,
              const MatrixXs& Sigma,
              std::map<ConstraintBasePtr,MatrixXs>& ctr_2_Phi)
{
    if (ctr_2_Phi.find(ctr_pair_it->first) == ctr_2_Phi.end())
        ctr_2_Phi[ctr_pair_it->first] = (J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()) *
                                         Sigma *
                                         J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).transpose()).inverse();
}

MatrixXs initOffDiagCorrector(const Eigen::MatrixXi& rows_and_cols)
{
    MatrixXs off_diag_corrector = MatrixXs::Identity(rows_and_cols.rows(),rows_and_cols.rows());
    for (auto i = 0; i<rows_and_cols.rows(); i++)
        if (rows_and_cols(i,0) != rows_and_cols(i,1))
            off_diag_corrector(i,i) = 2;

    return off_diag_corrector;
}

MatrixXs factorDescent(const MatrixXs& J, const MatrixXs& Sigma, const std::map<ConstraintBasePtr,unsigned int>& ctr_2_row, const PruningOptimOptions& options, const bool store_logs)
{
    clock_t t0 = clock();
    clock_t t1 = clock();
    std::vector<int> times, KLD_decrease;

    // build Omega
    unsigned int omega_size = J.rows();
    unsigned int sigma_size = Sigma.rows();
    MatrixXs Omega(omega_size,omega_size);
    for (auto ctr_pair : ctr_2_row)
        insertSparseBlock(ctr_pair.first->getFeaturePtr()->getMeasurementInformation(), Omega, ctr_pair.second, ctr_pair.second);

    // Indexes
    unsigned int N_steps = 0;
    std::map<ConstraintBasePtr,unsigned int> ctr_2_ind;
    Eigen::MatrixXi rows_and_cols = rowsAndCols(ctr_2_row, omega_size, ctr_2_ind);
    MatrixXs off_diag_corrector = initOffDiagCorrector(rows_and_cols);

    MatrixXs Upsilon_k(MatrixXs::Zero(sigma_size,sigma_size));
    std::map<ConstraintBasePtr,MatrixXs> ctr_2_Phi;
    std::map<ConstraintBasePtr,MatrixXs> ctr_2_QL;
    std::map<ConstraintBasePtr,MatrixXs> ctr_2_Q0;
    std::map<ConstraintBasePtr,MatrixXs> ctr_2_iL;
    int KLD;
    VectorXs KLD_gradient(rows_and_cols.rows());
    MatrixXs iLJt(ctr_2_row.begin()->first->getSize(), ctr_2_row.begin()->first->getSize());
    MatrixXs Omega_k(ctr_2_row.begin()->first->getSize(), ctr_2_row.begin()->first->getSize());
    MatrixXs Omega_k2(ctr_2_row.begin()->first->getSize(), ctr_2_row.begin()->first->getSize());
    MatrixXs prev_Omega_k(ctr_2_row.begin()->first->getSize(), ctr_2_row.begin()->first->getSize());
    MatrixXs prev_Omega = Omega;

    MatrixXs Ji(ctr_2_row.begin()->first->getSize(),sigma_size);
    Eigen::HouseholderQR<MatrixXs> qr(sigma_size,ctr_2_row.begin()->first->getSize());
    MatrixXs Q(sigma_size,sigma_size);
    MatrixXs R(ctr_2_row.begin()->first->getSize(),ctr_2_row.begin()->first->getSize());
    Eigen::ColPivHouseholderQR<MatrixXs> Upsilon_decomp(sigma_size,sigma_size);

    // Constant factors
    std::map<ConstraintBasePtr,bool> closed_form_factor;
    if (!options.apply_closed_form_)
        for (auto pair : ctr_2_row)
            closed_form_factor[pair.first] = false;

    // Times
    int t_init(0.0), t_upsilon(0.0), t_solve(0.0), t_pd(0.0), t_update(0.0), t_cost(0.0), t_checks(0.0);

    // initial
    int prev_KLD = 0.5*((J.transpose()*Omega*J*Sigma).trace() - logdet(J.transpose()*Omega*J*Sigma, std::string("factorDescent init")) - sigma_size);
    //std::cout << "init KLD = " << KLD << std::endl;

    // store logs
    if (store_logs)
    {
        times.push_back(0.0);
        KLD_decrease.push_back(prev_KLD);
    }

    // SPD factors
    std::map<ConstraintBasePtr,bool> ctr_2_spd;
    for (auto pair : ctr_2_row)
        ctr_2_spd[pair.first] = false;

    // KLD Gradient
    MatrixXs KLD_gradient_all = J * Sigma * J.transpose() - J * (J.transpose()*Omega*J).inverse() * J.transpose();
    Matrix2vector(KLD_gradient_all, KLD_gradient, rows_and_cols);
    KLD_gradient = off_diag_corrector * KLD_gradient;

    bool ending = false;
    // First pair
    // cyclic
    auto ctr_pair_it = ctr_2_row.begin();
    // non-cyclic
    if (options.cyclic_ != 0)
        getGradientDirection(KLD_gradient, ctr_2_row, ctr_2_ind, closed_form_factor, ctr_2_spd, ctr_pair_it);

    while (!ending)
    {
        if (closed_form_factor[ctr_pair_it->first])
            assert(false && "Closed form factor in FD iteration --> SHOULD NOT HAPPEN" );

        // ---------------------------------------------------------------------------------------------------
        // PRE-PROCESS ---------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------

        //std::cout << "init Phi\n";
        t1 = clock();
        initPhiK(ctr_pair_it, J, Sigma, ctr_2_Phi);
        t_init += ((int) clock() - t1) / CLOCKS_PER_SEC;

        //std::cout << "update Upsilon_k\n";
        t1 = clock();
        updateUpsilon(ctr_pair_it, J, Omega, omega_size, Upsilon_k);
        t_upsilon += ((int) clock() - t1) / CLOCKS_PER_SEC;

        // ---------------------------------------------------------------------------------------------------
        // SOLVING Omega_k -----------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------

        t1 = clock();
        computeOmegaK(ctr_pair_it, ctr_2_Phi, ctr_2_QL, ctr_2_Q0, ctr_2_iL, Ji, qr, Q, R, Upsilon_decomp, Upsilon_k, J, options, Omega_k, Omega_k2);
        t_solve += ((int) clock() - t1) / CLOCKS_PER_SEC;

        // PROPOSITION 3: Set closed form factor ========================================
        //std::cout << "Set closed form factor\n";
        if (options.apply_closed_form_ && closed_form_factor.find(ctr_pair_it->first) == closed_form_factor.end())
            closed_form_factor[ctr_pair_it->first] = Omega_k2.cwiseAbs().maxCoeff() < 1e-9;

        // ---------------------------------------------------------------------------------------------------
        // POST-PROCESS --------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------

        // No NaN's or Inf's
        if (Omega_k.allFinite())
        {
            // TODO Huge jumps stop half-way

            // make symmetric
            Omega_k = Omega_k.selfadjointView<Eigen::Upper>();

            // Positive definiteness
            //std::cout << "Positive definiteness\n";
            t1 = clock();
            if (!setSPD(Omega_k))
                ctr_2_spd[ctr_pair_it->first] = false;
            t_pd += ((int) clock() - t1) / CLOCKS_PER_SEC;
            t1 = clock();

            // Update Omega
            //std::cout << "Update Omega\n";
            assignSparseBlock(Omega_k,Omega,ctr_pair_it->second,ctr_pair_it->second);
            t_update += ((int) clock() - t1) / CLOCKS_PER_SEC;
            t1 = clock();

            // Update constraint
            ctr_pair_it->first->getFeaturePtr()->setMeasurementInformation(Omega_k);

            KLD = 0.5*((J.transpose()*Omega*J*Sigma).trace() - logdet(J.transpose()*Omega*J*Sigma, std::string("factorDescent"))-sigma_size);
            //std::cout << N_steps << "\tKLD = " << KLD << std::endl;
            t_cost += ((int) clock() - t1) / CLOCKS_PER_SEC;
            t1 = clock();

            // KLD Gradient
            //std::cout << "KLD Gradient\n";
            KLD_gradient_all = J * Sigma * J.transpose() - J * (J.transpose()*Omega*J).inverse() * J.transpose();
            Matrix2vector(KLD_gradient_all, KLD_gradient, rows_and_cols);
            KLD_gradient = off_diag_corrector * KLD_gradient;

            // check KLD improvement
            //std::cout << "check KLD improvement\n";
            if (KLD > prev_KLD)
            {
                //std::cout << "\tKO: Cost increased! KLD = " << KLD << " Gradient max coeff = " << std::max(KLD_gradient.maxCoeff(),-KLD_gradient.minCoeff()) << std::endl;
                //Omega = prev_Omega;
                prev_Omega_k = prev_Omega.block(ctr_pair_it->second,ctr_pair_it->second, ctr_pair_it->first->getSize(), ctr_pair_it->first->getSize());
                assignSparseBlock(prev_Omega_k,Omega,ctr_pair_it->second,ctr_pair_it->second);
                KLD = prev_KLD;

                // Restore constraint
                ctr_pair_it->first->getFeaturePtr()->setMeasurementInformation(prev_Omega_k);
            }
            // store prev values
            else
            {
                prev_KLD = KLD;
                //prev_Omega = Omega;
                assignSparseBlock(Omega_k,prev_Omega,ctr_pair_it->second,ctr_pair_it->second);
            }
            t_checks += ((int) clock() - t1) / CLOCKS_PER_SEC;

            //assert((Omega_k - Omega_k.transpose()).cwiseAbs().maxCoeff() < 1e-9 && "Omega_k is not symmetric!");
        }
        // Any NaN or Inf
        else
        {
            //std::cout << "Factor Descent ---------------- Any NaN or Inf in Omega_k!\n";
            //std::cout << "NaN in Omega_k: " << Omega_k.hasNaN() << std::endl;
            //std::cout << "NaN in first term: " << ctr_2_Phi.at(ctr_pair_it->first).hasNaN() << std::endl;
            //std::cout << "NaN in Jacobian: " << J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).hasNaN() << std::endl;
            //std::cout << "NaN in Upsilon: " << Upsilon_k.hasNaN() << std::endl;
            //std::cout << "Inf in Omega_k: " << !Omega_k.allFinite() << std::endl;
            //std::cout << "Inf in first term: " << !ctr_2_Phi.at(ctr_pair_it->first).allFinite() << std::endl;
            //std::cout << "Inf in Jacobian: " << !J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()).allFinite() << std::endl;
            //std::cout << "Inf in Upsilon: " << !Upsilon_k.allFinite() << std::endl;
            //std::cout << "Omega_k:\n" << Omega_k<< std::endl;
            //std::cout << "first term:\n" << ctr_2_Phi.at(ctr_pair_it->first) << std::endl;
            //std::cout << "Jacobian:\n" << J.middleRows(ctr_pair_it->second,ctr_pair_it->first->getSize()) << std::endl;
            //std::cout << "Upsilon_k:\n" << Upsilon_k << std::endl;
            KLD = prev_KLD;
        }

        // step
        N_steps++;

        // end conditions
        //std::cout << "end conditions\n";
        t1 = clock();
        // time
        if (((int) clock() - t0) / CLOCKS_PER_SEC > options.max_time_)
        {
            //std::cout << (options.cyclic_ == 0 ? "" : "nc") << "FD stopped for max " << options.max_time_ << "s constraint!" << std::endl;
            ending=true;
        }

        // steps
        if (N_steps == options.max_iter_)
            ending=true;

        // KLD Gradient
        if (KLD_gradient.cwiseAbs().maxCoeff() < options.min_gradient_)
            ending=true;

        if (ending)
            break;

        t_checks += ((int) clock() - t1) / CLOCKS_PER_SEC;

        // Next pair
        //std::cout << "Next pair\n";
        // cyclic
        if (options.cyclic_ == 0 )
        {
            auto current_pair_it = ctr_pair_it;
            do
            {
                ctr_pair_it++;
                if (ctr_pair_it == ctr_2_row.end())
                    ctr_pair_it = ctr_2_row.begin();
            }
            while (closed_form_factor[ctr_pair_it->first] && ctr_pair_it != current_pair_it);

            // all factors are constant --> end
            if (ctr_pair_it == current_pair_it)
                break;
        }
        // non-cyclic
        else
            getGradientDirection(KLD_gradient, ctr_2_row, ctr_2_ind, closed_form_factor, ctr_2_spd, ctr_pair_it);

        // store logs
        //std::cout << "store logs\n";
        if (store_logs)
        {
            times.push_back(((int) clock() - t0) / CLOCKS_PER_SEC);
            KLD_decrease.push_back(KLD);
        }
    }

    // store last logs
    if (store_logs)
    {
        times.push_back(((int) clock() - t0) / CLOCKS_PER_SEC);
        KLD_decrease.push_back(KLD);

        // convert to MatrixXs
        MatrixXs logs(2,times.size());
        logs.row(0) = Eigen::Map<VectorXs>(times.data(), times.size());
        logs.row(1) = Eigen::Map<VectorXs>(KLD_decrease.data(), KLD_decrease.size());
        return logs;
    }

//    KLD = 0.5*((J.transpose()*Omega*J*Sigma).trace() - logdet(J.transpose()*Omega*J*Sigma) - sigma_size));
//    std::cout << "Factor Descent " << (options.cyclic_==0 ? "cyclic" : "non-cyclic") << std::endl;
//    std::cout << "\tOmega size = " << Omega.rows() << std::endl;
//    std::cout << "\tKLD = " << KLD << std::endl;
//    std::cout << "\tsteps = " << N_steps << std::endl;
//    std::cout << "\tTotal time     = " << ((int) clock() - t0) / CLOCKS_PER_SEC << std::endl;
//    std::cout << "\ttime init      = " << t_init << std::endl;
//    std::cout << "\ttime upsilon   = " << t_upsilon << std::endl;
//    std::cout << "\ttime solve     = " << t_solve << std::endl;
//    std::cout << "\ttime PD        = " << t_pd << std::endl;
//    std::cout << "\ttime update    = " << t_update << std::endl;
//    std::cout << "\ttime cost      = " << t_cost << std::endl;
//    std::cout << "\ttime checks    = " << t_checks << std::endl;

    return MatrixXs::Constant(1,1,KLD);
}

#endif
