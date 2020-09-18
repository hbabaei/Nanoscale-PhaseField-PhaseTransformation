/* Author: Hamed Babaei, 2016 */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/std_cxx11/shared_ptr.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <iostream>
#include <fstream>
//////////////////////////////////////////////////////////////////
namespace PhaseField
{
  using namespace dealii;
// INPUT OF PARAMETERS
  namespace Parameters
  {
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");
        prm.declare_entry("Quadrature order", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }
////////////////////////////////////////////////////
    struct Geometry
    {
      unsigned int refinement;
      double       scale;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Global refinement", "4",
                           Patterns::Integer(0),
                           "Global refinement level");
        prm.declare_entry("Grid scale", "1",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");
      }
      prm.leave_subsection();
    }
    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        refinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
      }
      prm.leave_subsection();
    }

 /////////////////////////////////////////////////
        struct Materials
            {
              double lambda0; // austenite phase
              double mu0;     // austenite phase
              double lambda1;  // martensite phase
              double mu1;      // martensite phase
              double L;        // interface mobility
              double beta;     // gradient energy coefficient
              double A0;       // parameter for barrier height
              double theta;    // temperature
              double thetac;   // critical temperature
              double thetae;   // equilibrium temperature
              double thetan;   // Crank-Nicolson parameter

              static void
              declare_parameters(ParameterHandler &prm);
              void
              parse_parameters(ParameterHandler &prm);
            };
            void Materials::declare_parameters(ParameterHandler &prm)
            {
              prm.enter_subsection("Material properties");
              {
                prm.declare_entry("lambda austenite", "144.0", /*  */
                                  Patterns::Double(),
                                  "lambda austenite");

                prm.declare_entry("mu austenite", "74.0",
                                  Patterns::Double(0.0),
                                  "mu austenite");

                prm.declare_entry("lambda martensite", "379.0",
                                  Patterns::Double(),
                                  "lambda martensite");

                prm.declare_entry("mu martensite", "134.0",
                                  Patterns::Double(0.0),
                                  "mu martensite");

                prm.declare_entry("kinetic coeff", "2.6",
                                  Patterns::Double(0.0),
                                  "kinetic coeff");

                prm.declare_entry("energy coeff", "0.1",
                                  Patterns::Double(0.0),
                                  "energy coeff");

                prm.declare_entry("barrier height", "0.028",
                                  Patterns::Double(),
                                  "barrier height");

                prm.declare_entry("temperature", "50.0",
                                  Patterns::Double(),
                                  "temperature");

                prm.declare_entry("temperature crit", "-183.0",
                                  Patterns::Double(),
                                  "temperature crit");

                prm.declare_entry("temperature eql", "215.0",
                                  Patterns::Double(),
                                  "temperature eql");

                prm.declare_entry("crank-nicolson parameter", "0.5",
                                   Patterns::Double(),
                                  "crank-nicolson parameter");
              }
              prm.leave_subsection();
            }

            void Materials::parse_parameters(ParameterHandler &prm)
            {
              prm.enter_subsection("Material properties");
              {
                lambda0 = prm.get_double("lambda austenite");
                mu0 = prm.get_double("mu austenite");
                lambda1 = prm.get_double("lambda martensite");
                mu1 = prm.get_double("mu martensite");
                L = prm.get_double("kinetic coeff");
                beta = prm.get_double("energy coeff");
                A0 = prm.get_double("barrier height");
                theta = prm.get_double("temperature");
                thetac = prm.get_double("temperature crit");
                thetae = prm.get_double("temperature eql");
                thetan = prm.get_double("crank-nicolson parameter");
                 }
              prm.leave_subsection();
            }


    /////////////////////////////////////////////////
    struct Time
    {
      double delta_t;
      double end_time;
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "5",
                          Patterns::Double(),
                          "End time");
        prm.declare_entry("Time step size", "0.01",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }
    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }
 ///////////////////////////////////////////////////////
    struct AllParameters : public FESystem,
      public Geometry,
      public Materials,
      public Time
    {
      AllParameters(const std::string &input_file);
      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };
    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }
    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      Time::declare_parameters(prm);
    }
    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      Time::parse_parameters(prm);
    }
  }

 //  DEFINE SECOND ORDER IDENTITY, AND TWO FOURTH ORDER IDENTITY TENSORS
  template <int dim>
  class StandardTensors
  {
  public:
    static const SymmetricTensor<2, dim> I;
    static const SymmetricTensor<4, dim> IxI;
    static const SymmetricTensor<4, dim> II;
  };
  template <int dim>
  const SymmetricTensor<2, dim>
  StandardTensors<dim>::I = unit_symmetric_tensor<dim>();
  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::IxI = outer_product(I, I);
  template <int dim>
  const SymmetricTensor<4, dim>
  StandardTensors<dim>::II = identity_tensor<dim>();

// DEFINE TIME STEP, CURRENT TIME ETC.
  class Time
  {
  public:
    Time (const double time_end,
          const double delta_t)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      delta_t(delta_t),
      strainstep(0)

    {}

    virtual ~Time()
    {}
    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }

    unsigned int get_strainstep() const
    {
      return strainstep;
    }
    void strainincrement()
    {
      ++strainstep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
    unsigned int strainstep;
  };

//////////////////////////////////////////////////////////
    template <int dim>
     class Material_Constitutive
     {
     public:
       Material_Constitutive(const double A0,
                                                  const double theta,
                                                  const double thetac,
                                                  const double thetae)
         :
         //kinetic tensors
         det_F(1.0),
         Fe(Tensor<2, dim>()),
         ge(StandardTensors<dim>::I),
         Be(SymmetricTensor<2, dim>()),
         I1(0.0),
         Ge(StandardTensors<dim>::I),
         Ee(Tensor<2, dim>()),
         FeEe(Tensor<2, dim>()),
         EeEe(Tensor<2, dim>()),

         //elastic constants
         C0(Vector<double> (9)),
         C1(Vector<double> (9)),
         lambda0(Vector<double> (3)),
         lambda1(Vector<double> (3)),
         lambda (Vector<double> (3)),
         mu0(Vector<double> (3)),
         mu1(Vector<double> (3)),
         mu (Vector<double> (3)),
         nu0(Vector<double> (3)),
         nu1(Vector<double> (3)),
         nu (Vector<double> (3)),
         dnu (Vector<double> (3)),
         dmu (Vector<double> (3)),
         dlambda (Vector<double> (3)),
         ddnu (Vector<double> (3)),
         ddmu (Vector<double> (3)),
         ddlambda (Vector<double> (3)),

         //Phase-field constants
         A00(A0),
         theta1(theta),
         thetac1(thetac),
         thetae1(thetae)

       {}

       ~Material_Constitutive()
       {}

       void update_material_data (const Tensor<2, dim> &F, const Tensor<2, dim> &F_e,
                      const double phi, const double dphi,const double ddphi,
                      const double dphi_doublewell, const double ddphi_doublewell,
                      const double dphi_deltaG,const double ddphi_deltaG,
                      const double phi_elastic, const double dphi_elastic, const double ddphi_elastic,
                      const double A, const double deltaG)
       {

         // Kinetic tensors
         Fe=F_e;
         det_F = determinant(F);    // determinant of total F
         ge = symmetrize(Fe*transpose(Fe));  // ge = Fe.Fe^T
         Be = 0.5*(ge-StandardTensors<dim>::I);  // Be = 1/2(ge-I)
         I1 = trace(Be);                         // first invariant of Be
         Ge = symmetrize(transpose(Fe)*Fe);  //Ge=Fe^T.Fe
         Ee = 0.5*(Ge-StandardTensors<dim>::I);  //Ee=1/2(Ge-I)
         FeEe= Fe*Ee;
         EeEe= Ee*Ee;

        // Elastic constants for Orthotropic materials
        C0[0]= 167.5;//C0_11
         C0[1]= 167.5;//C0_22
         C0[2]= 167.5;//C0_33
         C0[3]=  80.1;//C0_44
         C0[4]=  80.1;//C0_55
         C0[5]=  80.1;//C0_66
         C0[6]=  65.0;//C0_12
         C0[7]=  65.0;//C0_13
         C0[8]=  65.0;//C0_23

         C1[0]= 174.76;//C1_11
         C1[1]= 136.68;//C1_22
         C1[2]= 174.76;//C1_33
         C1[3]=  60.24;//C1_44
         C1[4]=  42.22;//C1_55
         C1[5]=  60.24;//C1_66
         C1[6]=  68.00;//C1_12
         C1[7]= 102.00;//C1_13
         C1[8]=  68.00;//C1_23

         lambda0[0]= C0[0]+C0[8]+2*C0[3]-(C0[6]+C0[7]+2*C0[4]+2*C0[5]);
         lambda0[1]= C0[1]+C0[7]+2*C0[4]-(C0[6]+C0[8]+2*C0[3]+2*C0[5]);
         lambda0[2]= C0[2]+C0[6]+2*C0[5]-(C0[7]+C0[8]+2*C0[3]+2*C0[4]);

         lambda1[0]= C1[0]+C1[8]+2*C1[3]-(C1[6]+C1[7]+2*C1[4]+2*C1[5]);
         lambda1[1]= C1[1]+C1[7]+2*C1[4]-(C1[6]+C1[8]+2*C1[3]+2*C1[5]);
         lambda1[2]= C1[2]+C1[6]+2*C1[5]-(C1[7]+C1[8]+2*C1[3]+2*C1[4]);

         mu0[0]= 0.5*(C0[6]+C0[7]-C0[8]);
         mu0[1]= 0.5*(C0[6]+C0[8]-C0[7]);
         mu0[2]= 0.5*(C0[7]+C0[8]-C0[6]);

         mu1[0]= 0.5*(C1[6]+C1[7]-C1[8]);
         mu1[1]= 0.5*(C1[6]+C1[8]-C1[7]);
         mu1[2]= 0.5*(C1[7]+C1[8]-C1[6]);

         nu0[0]= 0.5*(C0[4]+C0[5]-C0[3]);
         nu0[1]= 0.5*(C0[3]+C0[5]-C0[4]);
         nu0[2]= 0.5*(C0[3]+C0[4]-C0[5]);

         nu1[0]= 0.5*(C1[4]+C1[5]-C1[3]);
         nu1[1]= 0.5*(C1[3]+C1[5]-C1[4]);
         nu1[2]= 0.5*(C1[3]+C1[4]-C1[5]);

         for (unsigned int n=0; n<3; ++n)
         {
         lambda[n] = lambda0[n]+(lambda1[n]-lambda0[n])*phi_elastic;
         mu[n] = mu0[n]+(mu1[n]-mu0[n])*phi_elastic;
         nu[n] = nu0[n]+(nu1[n]-nu0[n])*phi_elastic;

         dlambda[n] = (lambda1[n]-lambda0[n])*dphi_elastic;
         dmu[n] =     (mu1[n]-mu0[n])*dphi_elastic;
         dnu[n] =     (nu1[n]-nu0[n])*dphi_elastic;

         ddlambda[n] = (lambda1[n]-lambda0[n])*ddphi_elastic;
         ddmu[n] =     (mu1[n]-mu0[n])*ddphi_elastic;
         ddnu[n] =     (nu1[n]-nu0[n])*ddphi_elastic;
         }

//          deltaG = A00*(theta1-thetae1)/3.0;   // difference in thermal energy
//         A = A00*(theta1-thetac1);            // compute barrier height

         deltaG1= deltaG;//Si parameters
        A1= A;

        // derivatives of interpolation functions
        dphi1 = dphi;
        ddphi1 = ddphi;
        dphi_doublewell1 = dphi_doublewell;
        ddphi_doublewell1 = ddphi_doublewell;
        dphi_deltaG1=dphi_deltaG;
        ddphi_deltaG1=ddphi_deltaG;

           Assert(det_F > 0, ExcInternalError());
       }
       SymmetricTensor<2, dim> get_tau()    // tau is the Kirchhoff stress := J*Cauchy stress
       {
            return get_tau_kirchhoff();
       }
       SymmetricTensor<4, dim> get_Jc() const // comupte the fourth order modulus tensor in the deformed config.
       {
           return get_Jc_modulus();
       }
        double get_det_F() const    // determinant of deformation gradient tensor
       {
         return det_F;
       }
       double get_driving_force_noStress() const  // driving force excluding the transformational work
       {
         return get_driv_forc_noStress ();
       }
       double get_deri_driving_force_noStress() const  // driving force excluding the transformational work
       {
         return get_deri_driv_forc_noStress ();
       }
     protected:
       double det_F;
       double I1;
       Tensor<2, dim> Fe;
       SymmetricTensor<2, dim> ge;
       SymmetricTensor<2, dim> Be;

       SymmetricTensor<2, dim> Ge;
       Tensor<2, dim> Ee;
       Tensor<2, dim> FeEe;
       Tensor<2, dim> EeEe;
       Vector<double> C0;
       Vector<double> C1;
       Vector<double> lambda0;
       Vector<double> lambda1;
       Vector<double> lambda;
       Vector<double> mu0;
       Vector<double> mu1;
       Vector<double> mu;
       Vector<double> nu0;
       Vector<double> nu1;
       Vector<double> nu;
       Vector<double> dnu;
       Vector<double> dmu;
       Vector<double> dlambda;
       Vector<double> ddnu;
       Vector<double> ddmu;
       Vector<double> ddlambda;
       double A00;
       double theta1;
       double thetac1;
       double thetae1;
       double deltaG1;
       double A1;
       double dphi1;
       double ddphi1;
       double dphi_doublewell1;
       double ddphi_doublewell1;
       double dphi_deltaG1;
       double ddphi_deltaG1;
       double phi_elastic1;

        // compute the Kirchhoff stress
       SymmetricTensor<2, dim> get_tau_kirchhoff() const
       {
        SymmetricTensor<2, dim> kirchhoff_stress;
        for (unsigned int n=0; n<dim; ++n)
          for (unsigned int i=0; i<dim; ++i)
            for (unsigned int j=0; j<dim; ++j)

                kirchhoff_stress[i][j] += lambda[n]*Ee[n][n]*Fe[i][n]*Fe[j][n]+
                                          mu[n]*(I1*Fe[i][n]*Fe[j][n]+Ee[n][n]*ge[i][j])+
                                          2*nu[n]*(Fe[i][n]*FeEe[j][n]+FeEe[i][n]*Fe[j][n]);

       return kirchhoff_stress;
       }
// compute the modulus J*d_ijkl
       SymmetricTensor<4, dim> get_Jc_modulus() const
       {

       SymmetricTensor<4,dim> elasticityTensor;

        for (unsigned int n=0; n<dim; ++n)
          for (unsigned int i=0; i<dim; ++i)
            for (unsigned int j=0; j<dim; ++j)
              for (unsigned int k=0; k<dim; ++k)
                for (unsigned int l=0; l<dim; ++l)

                  elasticityTensor[i][j][k][l] +=
                     lambda[n]*Fe[i][n]*Fe[j][n]*Fe[k][n]*Fe[l][n]+
                         mu[n]*(Fe[i][n]*Fe[j][n]*ge[k][l]+ ge[i][j]*Fe[k][n]*Fe[l][n])+
                         nu[n]*(Fe[i][n]*ge[j][k]*Fe[l][n]+ Fe[j][n]*ge[i][k]*Fe[l][n]+
                                Fe[i][n]*ge[j][l]*Fe[k][n]+ Fe[j][n]*ge[i][l]*Fe[k][n]);


          return elasticityTensor;
       }

       // compute the driving force excluding the transformational work
       double get_driv_forc_noStress () const
       {
           double d_elastic_energy (0.0);
           for (unsigned int n=0; n<dim; ++n)
               d_elastic_energy += dlambda[n]*Ee[n][n]*Ee[n][n] + 2*dmu[n]*Ee[n][n]*I1 + 4*dnu[n]*EeEe[n][n];

        return det_F*(-d_elastic_energy -A1*dphi_doublewell1-deltaG1*dphi1);
       }
       double get_deri_driv_forc_noStress () const
       {
           double dd_elastic_energy (0.0);
           for (unsigned int n=0; n<dim; ++n)
              dd_elastic_energy += 2*ddlambda[n]*Ee[n][n]*Ee[n][n] + 4*ddmu[n]*Ee[n][n]*I1 + 8*ddnu[n]*EeEe[n][n];

        return  det_F*(-dd_elastic_energy -A1*ddphi_doublewell1-deltaG1*ddphi1);
       }
     };
//////////////////////////////////////////////////////////////////////////////////////////
// updates the quadrature point history

  template <int dim>
    class PointHistory
    {
    public:
      PointHistory()
        :
        material(NULL),
        F_inv(StandardTensors<dim>::I),
        tau(SymmetricTensor<2, dim>()),
        Jc(SymmetricTensor<4, dim>())
      {}

      virtual ~PointHistory()
      {
        delete material;
        material = NULL;
      }

      void setup_lqp (const Parameters::AllParameters &parameters)
      {
        material = new Material_Constitutive<dim>(parameters.A0,
        parameters.theta, parameters.thetac, parameters.thetae);

        update_values(Tensor<2, dim>(), double (), double (), double(), double(),double() );
      }

      void update_values (const Tensor<2, dim> &Grad_u_n,
                          const double new_eta,
                          const double old_eta,
                          const double eta_laplacians,
                          const double thetan,
                          const double beta)
      {

// updating order parameter
      double eta = thetan*new_eta +(1-thetan)*old_eta;

    // Interpolation functions and their derivatives for traditional interpolation function
    //(This function is not used in the current version of theory anymore)
      const double phi = 3*pow (eta, 2.0)-2*pow (eta, 3.0);   // interpolation function
      const double dphi = 6*eta-6*pow (eta, 2.0);  // derivative of interpolation fn
      const double ddphi = 6-12*eta;
  // Interpolation functions and their derivatives for  interpolation of double-well barriar
      const double dphi_doublewell = 2*eta-6*pow (eta, 2.0)+4*pow (eta, 3.0);
      const double ddphi_doublewell =2-12*eta+12*pow (eta, 2.0);
  // Interpolation functions and their derivatives for  interpolation of jump in thermal energy delta_G
      const double dphi_deltaG=6*eta-9.3*pow (eta, 2.0)+13.2*pow (eta, 3.0)-16.5*pow (eta, 4.0)+6.6*pow (eta, 5.0);
      const double ddphi_deltaG=6-18.6*eta+39.6*pow (eta, 2.0)-66*pow (eta, 3.0)+33*pow (eta, 4.0);
  // Interpolation functions and their derivatives for  interpolation of elastic constants
      const double phi_elastic= 10*pow(eta, 3.0)-15*pow(eta, 4.0)+6*pow(eta, 5.0);
      const double dphi_elastic= 30*pow(eta, 2.0)-60*pow(eta, 3.0)+30*pow(eta, 4.0);
      const double ddphi_elastic= 60*eta -180*pow(eta, 2.0)+120*pow(eta, 3.0);
 // Interpolation functions and their derivatives for  interpolation of transformation strain
 // Follwoing constants are obtaned after calibration of instability criteria with atomistic simulations
      double a3;
      double w3;
      double a1;
      double w1;
      double A;
      double deltaG;

      a3=  3.3528;
      w3= -2.6471;
      a1= 0.9211*a3;
      w1= 1.0406*w3;
      A= 12.4192;
      deltaG= 2;

      const double phi_1 = a1*pow (eta, 2.0)+(10-3*a1+w1)*pow(eta,3.0)+(3*a1-2*w1-15)*pow(eta,4.0)+(6-a1+w1)*pow(eta,5.0);
      const double dphi_1 = 2*a1*eta+3*(10-3*a1+w1)*pow(eta,2.0)+4*(3*a1-2*w1-15)*pow(eta,3.0)+5*(6-a1+w1)*pow(eta,4.0);
      const double ddphi_1 = 2*a1+6*(10-3*a1+w1)*eta+12*(3*a1-2*w1-15)*pow(eta,2.0)+20*(6-a1+w1)*pow(eta,3.0);

      const double phi_3 = a3*pow (eta, 2.0)+(10-3*a3+w3)*pow(eta,3.0)+(3*a3-2*w3-15)*pow(eta,4.0)+(6-a3+w3)*pow(eta,5.0);
      const double dphi_3 = 2*a3*eta+3*(10-3*a3+w3)*pow(eta,2.0)+4*(3*a3-2*w3-15)*pow(eta,3.0)+5*(6-a3+w3)*pow(eta,4.0);
      const double ddphi_3 = 2*a3+6*(10-3*a3+w3)*eta+12*(3*a3-2*w3-15)*pow(eta,2.0)+20*(6-a3+w3)*pow(eta,3.0);

      Tensor<2, dim> eps_t;  // transformation strain tensor

      eps_t[0][0] = 0.1753;
      eps_t[1][1] = -0.447;
      eps_t[2][2] = 0.1753;

      Tensor<2, dim> eps_t_phi;
      eps_t_phi[0][0] =  0.1753*phi_1;
      eps_t_phi[1][1] = -0.447 *phi_3;
      eps_t_phi[2][2] =  0.1753*phi_1;

      Tensor<2, dim> eps_t_dphi;
      eps_t_dphi[0][0] =  0.1753*dphi_1;
      eps_t_dphi[1][1] = -0.447 *dphi_3;
      eps_t_dphi[2][2] =  0.1753*dphi_1;

      Tensor<2, dim> eps_t_ddphi;
      eps_t_ddphi[0][0] =  0.1753*ddphi_1;
      eps_t_ddphi[1][1] = -0.447 *ddphi_3;
      eps_t_ddphi[2][2] =  0.1753*ddphi_1;

      Tensor<2, dim> dinverse_Ft;
      dinverse_Ft[0][0] = -0.1753*dphi_1/pow((1+0.1753*phi_1), 2.0);
      dinverse_Ft[1][1] =  0.447 *dphi_3/pow((1- 0.447*phi_3), 2.0);
      dinverse_Ft[2][2] = -0.1753*dphi_1/pow((1+0.1753*phi_1), 2.0);


      F = (Tensor<2, dim>(StandardTensors<dim>::I) +  Grad_u_n);  // total F

      Ft = (Tensor<2, dim>(StandardTensors<dim>::I) +  Tensor<2, dim>(eps_t_phi)); //transformation deformation gradient

      Fe = F * invert(Ft); // elastic part of deformation gradient
      material->update_material_data(F, Fe,
                                     phi, dphi, ddphi,
                                     dphi_doublewell, ddphi_doublewell,
                                     dphi_deltaG,ddphi_deltaG,
                                     phi_elastic, dphi_elastic, ddphi_elastic,
                                     A,deltaG);
      F_inv = invert(F);
      F_inv_tr = transpose(F_inv);
      tau = material->get_tau(); // extracting kirchhoff stress
      Jc = material->get_Jc();  // extracting J*d_ijkl
      driving_force_noStress = material->get_driving_force_noStress(); // extracting driving force with no stress
      deri_driving_force_noStress = material->get_deri_driving_force_noStress();

      const Tensor<2, dim> temp_tensor = F_inv*Tensor<2, dim>(tau);
      const Tensor<2, dim> temp_tensor1 = temp_tensor * Fe;
      const Tensor<2, dim> temp_tensor2 = Tensor<2, dim>(tau)* dinverse_Ft;

    get_driv_forc = scalar_product(temp_tensor1, eps_t_dphi) +
                      driving_force_noStress; // computing total driving force

      get_deri_driv_forc = scalar_product(temp_tensor2, eps_t_dphi)+
                           scalar_product(temp_tensor1, eps_t_ddphi) +
                           deri_driving_force_noStress;

      get_driv_forc_total=  get_driv_forc+beta* eta_laplacians;

      Assert(determinant(F_inv) > 0, ExcInternalError());
   //////////////////////////////////////////////////////////////////////////////
      }
      const Tensor<2, dim> &get_F() const
      {
        return F;
      }

      double get_det_F() const
      {
        return material->get_det_F();
      }

      const Tensor<2, dim> &get_F_inv() const
      {
        return F_inv;
      }

      const Tensor<2, dim> &get_F_inv_tr() const
      {
        return F_inv_tr;
      }

      const SymmetricTensor<2, dim> &get_tau() const
      {
        return tau;
      }

      const SymmetricTensor<4, dim> &get_Jc() const
      {
        return Jc;
      }

      double get_driving_force() const
      {
        return get_driv_forc;
      }
      double get_deri_driving_force() const
      {
        return get_deri_driv_forc;
      }

      double get_driving_force_total() const
      {
       return get_driv_forc_total;
      }
    private:
      Material_Constitutive<dim> *material;
      Tensor<2, dim> F;
      Tensor<2, dim> F_inv;
      Tensor<2, dim> F_inv_tr;
      Tensor<2, dim> Ft;
      SymmetricTensor<2, dim> tau;
      SymmetricTensor<4, dim> Jc;
      Tensor<2, dim> Fe;
      double driving_force_noStress;
      double deri_driving_force_noStress;
      double dphi;
      double ddphi;
      double get_driv_forc;
      double get_deri_driv_forc;
      double get_driv_forc_total;
    };
///////////////////////////////////////////////////////////////

  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file);

    virtual
    ~Solid();

    void
    run();

  private:

    void    make_grid();
    void    system_setup();
    void    assemble_system();
    void    make_constraints(const int &it_nr);
    void    solve_nonlinear_timestep();
    unsigned int    solve();
    void    assemble_system_eta();
    void    solve_nonlinear_timestep_eta();
    unsigned int    solve_eta();
    void    setup_qph();
    void    update_qph_incremental();
    void    output_results() const;
    void    output_resultant_stress();


    Parameters::AllParameters          parameters;

    double                           vol_reference; // volume of the reference config
    double                           vol_current;  // volume of the current config

    Time                             time;  // variable of type class 'Time
    MPI_Comm                         mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    ConditionalOStream               pcout;

    const unsigned int               degree; // degree of polynomial of shape functions
    const FESystem<dim>              fe; // fe object
    DoFHandler<dim>                  dof_handler; // we have used two dof_handler: one for mechanics another for order parameter
    const unsigned int               dofs_per_cell;   // no of dofs per cell for the mechanics problem
    const FEValuesExtractors::Vector u_fe;
    const QGauss<dim>                qf_cell;  // quadrature points in the cell
    const QGauss<dim - 1>            qf_face;  // quadrature points at the face
    const unsigned int               n_q_points;  // no of quadrature points in the cell
    const unsigned int               n_q_points_f; // no of quadrature points at the face
    ConstraintMatrix                 constraints;  // constraint object

    FE_DGQ<dim>                     history_fe;
    DoFHandler<dim>                    history_dof_handler;

    std::vector<PointHistory<dim> >  quadrature_point_history;

    IndexSet                         locally_owned_dofs;
    IndexSet                         locally_relevant_dofs;


    TrilinosWrappers::SparseMatrix               tangent_matrix;  // tangent stiffenss matrix
    TrilinosWrappers::MPI::Vector                system_rhs;  // system right hand side or residual of mechanics problem
    TrilinosWrappers::MPI::Vector                solution;  // solution vector for displacement
    TrilinosWrappers::MPI::Vector                solution_update; // another vector containing the displacement soln


    const unsigned int               degree_eta; // degree of polynomial for eta
    FE_Q<dim>                      fe_eta;  // fe object for eta
    DoFHandler<dim>                 dof_handler_eta; //another dof_handler for eta
    const unsigned int               dofs_per_cell_eta; // dof per eta cell
    const QGauss<dim>               qf_cell_eta;
    const unsigned int               n_q_points_eta;
    ConstraintMatrix                 constraints_eta;
    IndexSet                        locally_owned_dofs_eta;
    IndexSet                        locally_relevant_dofs_eta;

    PETScWrappers::MPI::SparseMatrix           mass_matrix;  // mass matrix out of Ginxburg-Landau eqn
    PETScWrappers::MPI::SparseMatrix           laplace_matrix; // Laplace matrix out of Ginxburg-Landau eqn
    PETScWrappers::MPI::SparseMatrix           system_matrix_eta;
    PETScWrappers::MPI::SparseMatrix           nl_matrix;
    PETScWrappers::MPI::SparseMatrix           tmp_matrix;
    PETScWrappers::MPI::Vector                nl_term;
    PETScWrappers::MPI::Vector                system_rhs_eta;
    PETScWrappers::MPI::Vector                solution_eta;  // solution vector for eta
    PETScWrappers::MPI::Vector                old_solution_eta;
    PETScWrappers::MPI::Vector                solution_update_eta;
    PETScWrappers::MPI::Vector                tmp_vector;   // a vector used for solving eta

    Vector<double>                   resultant_stress;
    Vector<double>                   static_stress;
    bool                            apply_strain;
    unsigned int                     load_step;

  };
/////////////////////////////////////////////////////////
  // defines the initial condition for the order parameter
           template <int dim>
           class InitialValues : public Function<dim>
           {
           public:
             InitialValues (const int &timestep )
              :
                Function<dim>(),
                time_step (timestep)
                {}
             virtual double value(const Point<dim>   &p,
                                   const unsigned int  /*component = 0*/) const;
           private:
             const int time_step ;
           };

           template <int dim>
           double InitialValues<dim>::value (const Point<dim>  &p,
                                      const unsigned int /*component*/) const
          {

       // To speed up formation of martensitic bands, random values between 0-0.01 is considered within two bands
     // and 0-0.001 outside of them
               if (((p[1]-p[0]<12) && (p[1]-p[0]> 8)) ||
                   ((p[1]-p[0]<-8) && (p[1]-p[0]>  -12)))

                return (rand()%100+0.0)*0.0001;
               else
                return (rand()%100+0.0)*0.00001;

          }

//////////////////////////////////////////////////
// constructor
  template <int dim>
  Solid<dim>::Solid(const std::string &input_file)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),

    parameters(input_file),
    time(parameters.end_time, parameters.delta_t),

    pcout (std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator)
           == 0)),

    degree(parameters.poly_degree),
    fe(FE_Q<dim>(parameters.poly_degree), dim),
    dof_handler(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(0),
    qf_cell(parameters.quad_order),
    qf_face(parameters.quad_order),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size()),

    degree_eta(parameters.poly_degree),
    fe_eta (parameters.poly_degree),
    dof_handler_eta (triangulation),
    dofs_per_cell_eta(fe_eta.dofs_per_cell),
    qf_cell_eta(parameters.quad_order),
    n_q_points_eta (qf_cell_eta.size()),

    history_dof_handler (triangulation),
    history_fe (parameters.poly_degree),
    apply_strain(false),
    load_step(1)
  {}

//destructor
  template <int dim>
  Solid<dim>::~Solid()
  {
    dof_handler.clear();
    dof_handler_eta.clear();
  }


///////////////////////////////////////////////////////
// Creating geometry and discritization
  template <int dim>
  void Solid<dim>::make_grid()
  {
/*
 // This way we can have different number of elements in different spacial directions
    std::vector< unsigned int > repetitions(dim, 25);
    if (dim == 3)
    repetitions[dim-1] = 25;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                 repetitions,
                                              Point<dim>(0.0, 0.0, 0.0),
                                               Point<dim>(20.0, 20.0, 20.0),
                                              true);
*/

      GridGenerator::hyper_rectangle(triangulation,
                                     Point<dim>(0.0, 0.0, 0.0),
                                     Point<dim>(20.0, 20.0, 20.0),
                                     true);
// Implementing periodic condition on the triangulation
      std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator> >
           periodicity_vector;
      GridTools::collect_periodic_faces(triangulation,
                                            /*b_id1*/ 0,
                                            /*b_id2*/ 1,
                                            /*direction*/ 0,
                                            periodicity_vector);
      GridTools::collect_periodic_faces(triangulation,
                                           /*b_id1*/ 2,
                                           /*b_id2*/ 3,
                                           /*direction*/ 1,
                                           periodicity_vector);
      GridTools::collect_periodic_faces(triangulation,
                                            /*b_id1*/ 4,
                                            /*b_id2*/ 5,
                                            /*direction*/ 2,
                                            periodicity_vector);
      triangulation.add_periodicity(periodicity_vector);

// mesh refinement
      triangulation.refine_global (parameters.refinement);



  }
///////////////////////////////////////////////////////
// Setup of system matrix and rhs with the DOFs
  template <int dim>
  void Solid<dim>::system_setup()
  {
    dof_handler.distribute_dofs(fe);
    dof_handler_eta.distribute_dofs (fe_eta);
    history_dof_handler.distribute_dofs (history_fe);

    const unsigned int n_dofs = dof_handler.n_dofs(),
                        n_dofs_eta  = dof_handler_eta.n_dofs();

         pcout     << "   Number of active cells: "
                       << triangulation.n_active_cells()
                       << std::endl
                       << "   Total number of cells: "
                       << triangulation.n_cells()
                       << std::endl
                          << "   Number of degrees of freedom: "
                       << n_dofs + n_dofs_eta
                       << " (" << n_dofs << '+' << n_dofs_eta << ')'
                       << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler  ,  locally_relevant_dofs);

    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    {

      DoFTools::make_periodicity_constraints(dof_handler,
                                             /*b_id*/ 0,
                                             /*b_id*/ 1,
                                             /*direction*/ 0,
                                             constraints);

      DoFTools::make_periodicity_constraints(dof_handler,
                                             /*b_id*/ 2,
                                             /*b_id*/ 3,
                                             /*direction*/ 1,
                                             constraints);

      DoFTools::make_periodicity_constraints(dof_handler,
                                               /*b_id*/ 4,
                                              /*b_id*/ 5,
                                               /*direction*/ 2,
                                              constraints);

   }
   constraints.close ();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);

    tangent_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);

    solution.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    solution_update.reinit(locally_owned_dofs,
                           locally_relevant_dofs,
                           mpi_communicator);

    system_rhs.reinit(locally_owned_dofs,
                      mpi_communicator);


    locally_owned_dofs_eta = dof_handler_eta.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler_eta  ,  locally_relevant_dofs_eta);

    constraints_eta.clear ();
    constraints_eta.reinit (locally_relevant_dofs_eta);
    {

        DoFTools::make_periodicity_constraints(dof_handler_eta,
                                               /*b_id*/ 0,
                                               /*b_id*/ 1,
                                               /*direction*/ 0,
                                               constraints_eta);
        DoFTools::make_periodicity_constraints(dof_handler_eta,
                                               /*b_id*/ 2,
                                               /*b_id*/ 3,
                                               /*direction*/ 1,
                                               constraints_eta);
        DoFTools::make_periodicity_constraints(dof_handler_eta,
                                               /*b_id*/ 4,
                                              /*b_id*/ 5,
                                              /*direction*/ 2,
                                               constraints_eta);

    }
    constraints_eta.close ();

    DynamicSparsityPattern dsp_eta(locally_relevant_dofs_eta);
    DoFTools::make_sparsity_pattern (dof_handler_eta, dsp_eta, constraints_eta, true);
    SparsityTools::distribute_sparsity_pattern (dsp_eta,
                                                dof_handler_eta.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs_eta);

    mass_matrix.reinit (locally_owned_dofs_eta,
                        locally_owned_dofs_eta,
                        dsp_eta,
                        mpi_communicator);
    laplace_matrix.reinit (locally_owned_dofs_eta,
                           locally_owned_dofs_eta,
                           dsp_eta,
                           mpi_communicator);
    system_matrix_eta.reinit (locally_owned_dofs_eta,
                              locally_owned_dofs_eta,
                              dsp_eta,
                              mpi_communicator);
    nl_matrix.reinit (locally_owned_dofs_eta,
                      locally_owned_dofs_eta,
                      dsp_eta,
                      mpi_communicator);
    tmp_matrix.reinit (locally_owned_dofs_eta,
                       locally_owned_dofs_eta,
                       dsp_eta,
                       mpi_communicator);

    solution_eta.reinit(locally_owned_dofs_eta,
                        locally_relevant_dofs_eta,
                        mpi_communicator);
    old_solution_eta.reinit(locally_owned_dofs_eta,
                            locally_relevant_dofs_eta,
                            mpi_communicator);
    solution_update_eta.reinit(locally_owned_dofs_eta,
                               locally_relevant_dofs_eta,
                               mpi_communicator);

    system_rhs_eta.reinit(locally_owned_dofs_eta,
                          mpi_communicator);
    nl_term.reinit(locally_owned_dofs_eta,
                   mpi_communicator);
    tmp_vector.reinit(locally_owned_dofs_eta,
                      mpi_communicator);

    resultant_stress.reinit(1000);

    setup_qph();

  }

  //////////////////////////////////
// Applying displacement constraints and boundary conditions
  template <int dim>
  void Solid<dim>::make_constraints(const int &it_nr)
  {
    if (it_nr > 1)
      return;

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);

    const bool apply_dirichlet_bc = (it_nr == 0);
    const int  timestep = time.get_timestep();

    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    const FEValuesExtractors::Scalar z_displacement(2);

// Fixing points or lines
        const double tol_boundary = 0.1;
        typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
              {
                  for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)

                  if     ((std::abs(cell->vertex(v)[0] -10.0) < tol_boundary) &&
                          (std::abs(cell->vertex(v)[1] -10.0) < tol_boundary) &&
                          (std::abs(cell->vertex(v)[2] -10.0) < tol_boundary))
                       {
                          constraints.add_line(cell->vertex_dof_index(v, 0));
                          constraints.add_line(cell->vertex_dof_index(v, 1));
                          constraints.add_line(cell->vertex_dof_index(v, 2));
                       }


              }
// Fixing external surfaces
/*
     {
        const int boundary_id = 4;

         if (apply_dirichlet_bc == true)
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   boundary_id,
                                                   ZeroFunction<dim>(dim),
                                                   constraints,
                                                   fe.component_mask(z_displacement));
        else
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   boundary_id,
                                                   ZeroFunction<dim>(dim),
                                                   constraints,
                                                   fe.component_mask(z_displacement));
      }

      {
        const int boundary_id = 5;

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 ZeroFunction<dim>(dim),
                                                 constraints,
                                                 fe.component_mask(z_displacement));
      else
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_id,
                                                 ZeroFunction<dim>(dim),
                                                 constraints,
                                                 fe.component_mask(z_displacement));
      }
*/

// periodic related constraints
      {

        DoFTools::make_periodicity_constraints(dof_handler,
                                              /*b_id*/ 0,
                                              /*b_id*/ 1,
                                              /*direction*/ 0,
                                              constraints);
        DoFTools::make_periodicity_constraints(dof_handler,
                                              /*b_id*/ 2,
                                              /*b_id*/ 3,
                                              /*direction*/ 1,
                                               constraints);
       DoFTools::make_periodicity_constraints(dof_handler,
                                               /*b_id*/ 4,
                                               /*b_id*/ 5,
                                               /*direction*/ 2,
                                               constraints);
      }

     constraints.close();

// boundary displacement on periodic faces
     {
         IndexSet selected_dofs_x;
         std::set< types::boundary_id > boundary_ids_x= std::set<types::boundary_id>();
                 boundary_ids_x.insert(0);

         DoFTools::extract_boundary_dofs(dof_handler,
                                        fe.component_mask(x_displacement),
                                             selected_dofs_x,
                                             boundary_ids_x);
         unsigned int nb_dofs_face_x = selected_dofs_x.n_elements();
         IndexSet::ElementIterator dofs_x = selected_dofs_x.begin();

        double relative_displacement_x;
         if (timestep <3)
             relative_displacement_x = -0.23;
         else
             relative_displacement_x = -0.0056;

         for(unsigned int i = 0; i < nb_dofs_face_x; i++)
        {
           constraints.set_inhomogeneity(*dofs_x, (apply_dirichlet_bc ? relative_displacement_x : 0.0));
             dofs_x++;
         }
       }

     {
                  IndexSet selected_dofs_y;
                  std::set< types::boundary_id > boundary_ids_y= std::set<types::boundary_id>();
                          boundary_ids_y.insert(2);

                  DoFTools::extract_boundary_dofs(dof_handler,
                                                 fe.component_mask(y_displacement),
                                                    selected_dofs_y,
                                                    boundary_ids_y);
                  unsigned int nb_dofs_face_y = selected_dofs_y.n_elements();
                  IndexSet::ElementIterator dofs_y = selected_dofs_y.begin();

                  double relative_displacement_y;
                  if (timestep <3 )
                    relative_displacement_y = 0.82;
                 // else if (timestep <230 )
                   // relative_displacement_y = 0.02;
                  else
                    relative_displacement_y = 0.02;


                  for(unsigned int i = 0; i < nb_dofs_face_y; i++)
                  {

                    constraints.set_inhomogeneity(*dofs_y,(apply_dirichlet_bc ? relative_displacement_y : 0.0) );
                      dofs_y++;
                  }

                }


            {
                IndexSet selected_dofs_z;
                std::set< types::boundary_id > boundary_ids_z= std::set<types::boundary_id>();
                        boundary_ids_z.insert(4);

                DoFTools::extract_boundary_dofs(dof_handler,
                                               fe.component_mask(z_displacement),
                                                 selected_dofs_z,
                                                 boundary_ids_z);
                unsigned int nb_dofs_face_z = selected_dofs_z.n_elements();
                IndexSet::ElementIterator dofs_z = selected_dofs_z.begin();

                double relative_displacement_z;
                if (timestep <3)
                  relative_displacement_z = -0.23;
                else
                    relative_displacement_z = -0.0056;

                for(unsigned int i = 0; i < nb_dofs_face_z; i++)
                {

                  constraints.set_inhomogeneity(*dofs_z,(apply_dirichlet_bc ? relative_displacement_z : 0.0));
                    dofs_z++;
                }
              }


    constraints.close();

  }
  /////////////////////////////////////////////////////////////////////////////////////
// Assempling system matrix and rhs for mechanical equilibrium equation
template <int dim>
  void Solid<dim>::assemble_system ()
  {
    tangent_matrix = 0;
    system_rhs = 0;

   FEValues<dim> fe_values (fe, qf_cell,
                            update_values   | update_gradients |
                            update_quadrature_points | update_JxW_values);


   FEFaceValues<dim> fe_face_values (fe, qf_face,
                                     update_values         | update_quadrature_points  |
                                     update_normal_vectors | update_JxW_values);

   FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
   Vector<double>       cell_rhs (dofs_per_cell);

   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

   std::vector<double>                    Nx(dofs_per_cell);
   std::vector<Tensor<2, dim> >           grad_Nx(dofs_per_cell);
   std::vector<SymmetricTensor<2, dim> >  symm_grad_Nx(dofs_per_cell);



   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   for (; cell!=endc;  ++cell)
       if (cell->is_locally_owned())
       {
           fe_values.reinit (cell);
           cell_matrix = 0;
        cell_rhs = 0;

       PointHistory<dim> *lqph =
             reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
           {
            const Tensor<2, dim> F_inv = lqph[q_point].get_F_inv();
            const Tensor<2, dim> tau         = lqph[q_point].get_tau();
            const SymmetricTensor<2, dim> symm_tau         = lqph[q_point].get_tau();
            const SymmetricTensor<4, dim> Jc = lqph[q_point].get_Jc();
            const double JxW = fe_values.JxW(q_point);

            for (unsigned int k=0; k<dofs_per_cell; ++k)
               {
                  grad_Nx[k] = fe_values[u_fe].gradient(k, q_point)  * F_inv;
                  symm_grad_Nx[k] = symmetrize(grad_Nx[k]);
               }


          for (unsigned int i=0; i<dofs_per_cell; ++i)
             {
               const unsigned int component_i = fe.system_to_component_index(i).first;

                for (unsigned int j=0; j<dofs_per_cell; ++j)
               {
                     const unsigned int component_j = fe.system_to_component_index(j).first;

                      cell_matrix(i, j) += symm_grad_Nx[i] * Jc // The material contribution:
                                                  * symm_grad_Nx[j] * JxW;
                     if (component_i == component_j) // geometrical stress contribution
                      cell_matrix(i, j) += grad_Nx[i][component_i] * tau
                                                  * grad_Nx[j][component_j] * JxW;
              }

                      cell_rhs(i) -= symm_grad_Nx[i] * symm_tau * JxW;
            }
       }

// Applying stress on the external faces in the case of stress-controlled loading
/*
        if (time.get_timestep()>230)
        {

         for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)

             if (cell->face(face_number)->at_boundary()
                &&
                (cell->face(face_number)->boundary_id() == 0
                  ||
                 cell->face(face_number)->boundary_id() == 1
                  ||
                 cell->face(face_number)->boundary_id() == 4
                  ||
                 cell->face(face_number)->boundary_id() == 5))
              {
                fe_face_values.reinit (cell, face_number);

                for (unsigned int q_point=0; q_point<n_q_points_f; ++q_point)
                  {
                    const Tensor<2, dim> F_inv_tr = lqph[q_point].get_F_inv_tr();

                   // const Tensor<1, dim> current_normal_dir = (F_inv_tr*fe_face_values.normal_vector(q_point))/
                   //                                           (F_inv_tr*fe_face_values.normal_vector(q_point)).norm() ;
                    const double J= lqph[q_point].get_det_F();

                    double magnitude;
                     //magnitude=5.0e9 * J * (F_inv_tr*fe_face_values.normal_vector(q_point)).norm();
                         magnitude = load_step * J * (F_inv_tr*fe_face_values.normal_vector(q_point)).norm();;

                   const Tensor<1, dim> traction  = magnitude* fe_face_values.normal_vector(q_point);


                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        const unsigned int
                      component_i = fe.system_to_component_index(i).first;
                      cell_rhs(i) += (traction[component_i] *
                                      fe_face_values.shape_value(i,q_point) *
                                      fe_face_values.JxW(q_point));
                    }
                  }
                }
             }

*/
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              tangent_matrix,
                                              system_rhs);

     }

   tangent_matrix.compress (VectorOperation::add);
   system_rhs.compress (VectorOperation::add);

  }

  ///////////////////////////////////////////////////////
// Assembling system matrix and rhs for Quinzburg-Landau equation
  template <int dim>
  void Solid<dim>::assemble_system_eta ()
  {
    system_matrix_eta = 0;
    system_rhs_eta = 0;

    mass_matrix = 0;
    laplace_matrix = 0;
    nl_matrix = 0;
    nl_term = 0;

    FEValues<dim> fe_values_eta (fe_eta, qf_cell_eta,
                             update_values  | update_gradients |
                             update_quadrature_points | update_JxW_values);

    FullMatrix<double>   cell_matrix_eta      (dofs_per_cell_eta, dofs_per_cell_eta);
    Vector    <double>   cell_rhs_eta         (dofs_per_cell_eta);

    FullMatrix<double>   cell_mass_matrix            (dofs_per_cell_eta, dofs_per_cell_eta);
    FullMatrix<double>   cell_laplace_matrix         (dofs_per_cell_eta, dofs_per_cell_eta);
    FullMatrix<double>   cell_nl_matrix              (dofs_per_cell_eta, dofs_per_cell_eta);
    Vector    <double>   cell_nl_term                (dofs_per_cell_eta);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_eta);

    std::vector<double> phi (dofs_per_cell_eta);
    std::vector<Tensor<1,dim> >   grad_phi (dofs_per_cell_eta);

    std::vector<double> solution_eta_total (qf_cell.size());
    std::vector<double> old_solution_eta_total (qf_cell.size());

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_eta.begin_active(),
    endc = dof_handler_eta.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        fe_values_eta.reinit(cell);
        cell_matrix_eta=0;
        cell_rhs_eta=0;

        cell_mass_matrix = 0;
        cell_laplace_matrix = 0;
        cell_nl_matrix = 0;
        cell_nl_term = 0;

        PointHistory<dim> *lqph =
                      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

        fe_values_eta.get_function_values(solution_eta,   solution_eta_total);
        fe_values_eta.get_function_values(old_solution_eta,   old_solution_eta_total);

            for (unsigned int q=0; q<n_q_points_eta; ++q)
          {
                const double driving_force = lqph[q].get_driving_force();
                const double deri_driving_force = lqph[q].get_deri_driving_force();

            for (unsigned int k=0; k<dofs_per_cell_eta; ++k)
              {
                phi[k]      = fe_values_eta.shape_value (k, q);
                grad_phi[k] = fe_values_eta.shape_grad  (k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell_eta; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell_eta; ++j)
                {

                    cell_matrix_eta(i,j) +=
                            ((phi[i] * phi[j]) +
                            parameters.L * parameters.beta * parameters.delta_t * parameters.thetan*(grad_phi[i] * grad_phi[j]) -
                            parameters.L * parameters.delta_t * parameters.thetan*(deri_driving_force * phi[i] * phi[j]))
                            *fe_values_eta.JxW(q);


                    cell_mass_matrix(i,j)    += (phi[i]      * phi[j])      * fe_values_eta.JxW(q);
                    cell_laplace_matrix(i,j) += (grad_phi[i] * grad_phi[j]) * fe_values_eta.JxW(q);
                    cell_nl_matrix(i,j) += deri_driving_force * phi[i] * phi[j] * fe_values_eta.JxW(q);
               }


                    cell_nl_term(i)     +=  driving_force * phi[i] * fe_values_eta.JxW (q);

            }
          }


        cell->get_dof_indices (local_dof_indices);
        constraints_eta.distribute_local_to_global (cell_matrix_eta,
                                                    local_dof_indices,
                                                    system_matrix_eta);

           for (unsigned int i=0; i<dofs_per_cell_eta; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell_eta; ++j)
              {
                mass_matrix.  add (local_dof_indices[i],
                                     local_dof_indices[j],
                                     cell_mass_matrix(i,j));

                laplace_matrix.add (local_dof_indices[i],
                                    local_dof_indices[j],
                                     cell_laplace_matrix(i,j));

                nl_matrix.     add (local_dof_indices[i],
                                       local_dof_indices[j],
                                       cell_nl_matrix(i,j));

              }
                nl_term (local_dof_indices[i]) += cell_nl_term(i);
          }
     }

    mass_matrix.compress (VectorOperation::add);
    laplace_matrix.compress (VectorOperation::add);
    nl_matrix.compress (VectorOperation::add);
    nl_term.compress (VectorOperation::add);

    PETScWrappers::MPI::Vector    temp_solution_eta (locally_owned_dofs_eta, mpi_communicator);
    PETScWrappers::MPI::Vector    temp_old_solution_eta (locally_owned_dofs_eta, mpi_communicator);
    temp_solution_eta = solution_eta;
    temp_old_solution_eta = old_solution_eta;

    PETScWrappers::MPI::Vector    uncondensed_system_rhs_eta (locally_owned_dofs_eta, mpi_communicator);
    PETScWrappers::MPI::Vector    ghosted_uncondensed_system_rhs_eta (locally_owned_dofs_eta,locally_relevant_dofs_eta, mpi_communicator);

    tmp_matrix.copy_from (mass_matrix);
    tmp_matrix.add (   laplace_matrix,parameters.L * parameters.beta * parameters.delta_t * parameters.thetan);

    tmp_matrix.vmult (tmp_vector, temp_solution_eta);
    uncondensed_system_rhs_eta += tmp_vector;

    tmp_matrix.add(laplace_matrix, -parameters.L * parameters.beta * parameters.delta_t);

    tmp_matrix.vmult (tmp_vector, temp_old_solution_eta);
    uncondensed_system_rhs_eta -= tmp_vector;
    uncondensed_system_rhs_eta.add (-parameters.L * parameters.delta_t, nl_term);

    uncondensed_system_rhs_eta *= -1;

    ghosted_uncondensed_system_rhs_eta= uncondensed_system_rhs_eta;
    constraints_eta.condense (ghosted_uncondensed_system_rhs_eta, system_rhs_eta);

    system_matrix_eta.compress (VectorOperation::add);
    system_rhs_eta.compress (VectorOperation::add);


  }

///////////////////////////////////////////////////////////////////////
// Setuping quadrature point history
  template <int dim>
  void Solid<dim>::setup_qph()
  {

    {
    unsigned int our_cells = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned())
        ++our_cells;
    triangulation.clear_user_data();
    {
      std::vector<PointHistory<dim> > tmp;
      tmp.swap (quadrature_point_history);
    }
    quadrature_point_history.resize (our_cells * n_q_points);

      unsigned int history_index = 0;
      for (typename Triangulation<dim>::active_cell_iterator
              cell = triangulation.begin_active();
              cell != triangulation.end(); ++cell)
       if (cell->is_locally_owned())
        {
          cell->set_user_pointer(&quadrature_point_history[history_index]);
          history_index += n_q_points;
        }

      Assert(history_index == quadrature_point_history.size(),
             ExcInternalError());
    }

    for (typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active();
            cell != triangulation.end(); ++cell)
     if (cell->is_locally_owned())
      {
        PointHistory<dim> *lqph =
          reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

        Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
        Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point].setup_lqp(parameters);
      }
  }
//////////////////////////////////////
// Updates the data at all quadrature points over a loop run
  template <int dim>
  void Solid<dim>::update_qph_incremental()

  {
    FEValues<dim> fe_values (fe, qf_cell,
                                 update_values | update_gradients);
    FEValues<dim> fe_values_eta (fe_eta, qf_cell,
                                     update_values | update_gradients| update_hessians);

    std::vector<Tensor<2, dim> > solution_grads_total (qf_cell.size());
    std::vector<double> solution_eta_total (qf_cell.size());
    std::vector<double> old_solution_eta_total (qf_cell.size());
    std::vector<double> solution_eta_laplacians (qf_cell.size());

    TrilinosWrappers::MPI::Vector       distributed_solution(locally_owned_dofs, mpi_communicator);
    PETScWrappers::MPI::Vector       distributed_solution_eta(locally_owned_dofs_eta, mpi_communicator);
    PETScWrappers::MPI::Vector       distributed_old_solution_eta(locally_owned_dofs_eta, mpi_communicator);
    distributed_solution = solution;
    distributed_solution_eta = solution_eta;
    distributed_old_solution_eta = old_solution_eta;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    typename DoFHandler<dim>::active_cell_iterator
    cell_eta = dof_handler_eta.begin_active();
    for (; cell!=endc; ++cell, ++cell_eta)
        if (cell->is_locally_owned())
        {
            PointHistory<dim> *lqph =
              reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

            Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
            Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

            Assert(solution_grads_total.size() == n_q_points,
                   ExcInternalError());
            Assert(solution_eta_total.size() == n_q_points,
                   ExcInternalError());

            fe_values.reinit(cell);
            fe_values_eta.reinit (cell_eta);

            fe_values[u_fe].get_function_gradients(solution,  solution_grads_total);
            fe_values_eta.get_function_values(solution_eta,   solution_eta_total);
            fe_values_eta.get_function_values(old_solution_eta,   old_solution_eta_total);
            fe_values_eta.get_function_laplacians(solution_eta,   solution_eta_laplacians);

           for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  lqph[q_point].update_values(solution_grads_total[q_point],
                                              solution_eta_total[q_point],
                                              old_solution_eta_total[q_point],
                                              solution_eta_laplacians[q_point],
                                              parameters.thetan,
                                              parameters.beta);
       }

  }
  //////////////////////////////
// Solving mechanical equilibrium equation
  template <int dim>
  unsigned int
  Solid<dim>::solve ()
    {

      TrilinosWrappers::MPI::Vector
      completely_distributed_solution (locally_owned_dofs, mpi_communicator);

      SolverControl solver_control (dof_handler.n_dofs(), 1e-8*system_rhs.l2_norm());

      TrilinosWrappers::SolverCG solver(solver_control);


      TrilinosWrappers::PreconditionAMG preconditioner;
      preconditioner.initialize(tangent_matrix);
      solver.solve (tangent_matrix, completely_distributed_solution, system_rhs,
                          preconditioner);

//      TrilinosWrappers::SolverDirect::AdditionalData data;
//      data.solver_type= "Amesos_Superludist";
//      TrilinosWrappers::SolverDirect solver (solver_control, data);
//      solver.solve (tangent_matrix, completely_distributed_solution, system_rhs);

      constraints.distribute (completely_distributed_solution);

      solution_update = completely_distributed_solution;

      return solver_control.last_step();
    }

///////////////////////////////////////////////////////////////////////////
// Solving Quinzburg-Landau equation
  template <int dim>
  unsigned int
  Solid<dim>::solve_eta ()
   {
      PETScWrappers::MPI::Vector
      completely_distributed_solution_eta (locally_owned_dofs_eta, mpi_communicator);

      SolverControl solver_control (dof_handler_eta.n_dofs(), 1e-12*system_rhs_eta.l2_norm());

      PETScWrappers::SolverCG solver(solver_control);

      PETScWrappers::PreconditionBoomerAMG preconditioner;
      preconditioner.initialize(system_matrix_eta);

      solver.solve (system_matrix_eta, completely_distributed_solution_eta, system_rhs_eta,
                    preconditioner);

      //constraints_eta.distribute (completely_distributed_solution_eta);

      solution_update_eta = completely_distributed_solution_eta;

      return solver_control.last_step();
   }
///////////////////////////////////////////////////////////
// Outputing displacemenr, order parameter and stress componenets
  template <int dim>
  void Solid<dim>::output_results() const
  {
    DataOut<dim> data_out;

    std::vector<std::string> displacement_names;
         switch (dim)
           {
           case 1:
             displacement_names.push_back ("displacement");
               break;
           case 2:
               displacement_names.push_back ("x_displacement");
               displacement_names.push_back ("y_displacement");
               break;
           case 3:
               displacement_names.push_back ("x_displacement");
               displacement_names.push_back ("y_displacement");
               displacement_names.push_back ("z_displacement");
               break;
          default:
               Assert (false, ExcNotImplemented());
          }

     data_out.add_data_vector (dof_handler, solution, displacement_names);
     data_out.add_data_vector (dof_handler_eta, solution_eta, "order_parameter");

     /////////////////////////
    Vector<double> norm_of_stress (triangulation.n_active_cells());

     {
       typename Triangulation<dim>::active_cell_iterator
       cell = triangulation.begin_active(),
       endc = triangulation.end();
       for (; cell!=endc; ++cell)
           if (cell->is_locally_owned())
           {
             SymmetricTensor<2,dim> accumulated_stress;
             for (unsigned int q=0; q<qf_cell.size(); ++q)
               accumulated_stress +=
                 reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].get_tau();
             norm_of_stress(cell->active_cell_index())
               = (accumulated_stress /
                  qf_cell.size()).norm();
           }
           else
            norm_of_stress(cell->active_cell_index()) = -1e+20;
      }
data_out.add_data_vector (norm_of_stress, "norm_of_stress");
///////////////////////////////////////////////
std::vector< std::vector< Vector<double> > >
     history_field_stress (dim, std::vector< Vector<double> >(dim)),
     local_history_values_at_qpoints_stress (dim, std::vector< Vector<double> >(dim)),
     local_history_fe_values_stress (dim, std::vector< Vector<double> >(dim));

   for (unsigned int i=0; i<dim; i++)
     for (unsigned int j=0; j<dim; j++)
     {
       history_field_stress[i][j].reinit(history_dof_handler.n_dofs());
       local_history_values_at_qpoints_stress[i][j].reinit(qf_cell.size());
       local_history_fe_values_stress[i][j].reinit(history_fe.dofs_per_cell);
     }

   Vector<double> history_field_drivingforce,
                  local_history_values_at_qpoints_drivingforce,
                  local_history_fe_values_drivingforce;

   history_field_drivingforce.reinit(history_dof_handler.n_dofs());
   local_history_values_at_qpoints_drivingforce.reinit(qf_cell.size());
   local_history_fe_values_drivingforce.reinit(history_fe.dofs_per_cell);

   Vector<double> history_field_drivingforce_total,
       local_history_values_at_qpoints_drivingforce_total,
       local_history_fe_values_drivingforce_total;

  history_field_drivingforce_total.reinit(history_dof_handler.n_dofs());
  local_history_values_at_qpoints_drivingforce_total.reinit(qf_cell.size());
  local_history_fe_values_drivingforce_total.reinit(history_fe.dofs_per_cell);

   FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
                                               qf_cell.size());
   FETools::compute_projection_from_quadrature_points_matrix
             (history_fe,
                 qf_cell, qf_cell,
              qpoint_to_dof_matrix);

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end(),
   dg_cell = history_dof_handler.begin_active();
   for (; cell!=endc; ++cell, ++dg_cell)
     if (cell->is_locally_owned())
     {
       PointHistory<dim> *lqph
              = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
       Assert (lqph >=
                   &quadrature_point_history.front(),
                   ExcInternalError());
       Assert (lqph <
                   &quadrature_point_history.back(),
                   ExcInternalError());
       for (unsigned int i=0; i<dim; i++)
         for (unsigned int j=0; j<dim; j++)
         {
           for (unsigned int q=0; q<qf_cell.size(); ++q)
             {

              local_history_values_at_qpoints_stress[i][j](q)
                                         = (lqph[q].get_tau()[i][j])/(lqph[q].get_det_F());
              qpoint_to_dof_matrix.vmult (local_history_fe_values_stress[i][j],
                                          local_history_values_at_qpoints_stress[i][j]);
              dg_cell->set_dof_values (local_history_fe_values_stress[i][j],
                                       history_field_stress[i][j]);

              local_history_values_at_qpoints_drivingforce(q)
                                         = lqph[q].get_driving_force();
              qpoint_to_dof_matrix.vmult (local_history_fe_values_drivingforce,
                                          local_history_values_at_qpoints_drivingforce);
              dg_cell->set_dof_values (local_history_fe_values_drivingforce,
                                       history_field_drivingforce);

              local_history_values_at_qpoints_drivingforce_total(q)
                                        = lqph[q].get_driving_force_total();
              qpoint_to_dof_matrix.vmult (local_history_fe_values_drivingforce_total,
                                          local_history_values_at_qpoints_drivingforce_total);
              dg_cell->set_dof_values (local_history_fe_values_drivingforce_total,
                                       history_field_drivingforce_total);

            }
        }
     }

   std::vector<DataComponentInterpretation::DataComponentInterpretation>
                 data_component_interpretation2(1, DataComponentInterpretation::component_is_scalar);


   data_out.add_data_vector(history_dof_handler, history_field_stress[0][0], "sigma_11",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[1][1], "sigma_22",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[2][2], "sigma_33",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[0][1], "sigma_12",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[0][2], "sigma_13",
                                data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_stress[1][2], "sigma_23",
                                data_component_interpretation2);


   data_out.add_data_vector(history_dof_handler, history_field_drivingforce, "local_driving_force",
                                   data_component_interpretation2);
   data_out.add_data_vector(history_dof_handler, history_field_drivingforce_total, "driving_force_total",
                                  data_component_interpretation2);
     //////////////////////////

    MappingQEulerian<dim, TrilinosWrappers::MPI::Vector > q_mapping(degree, dof_handler, solution);

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches(q_mapping, degree);

    const unsigned int cycle = time.get_timestep();

    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
         filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 2) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");
         std::ofstream master_output (("solution-" +
                                      Utilities::int_to_string (cycle, 2) +
                                      ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }

   }

  //////////////////////////////////
// Outputing average stress and strain over the external faces
  template <int dim>
    void Solid<dim>::output_resultant_stress()
    {

        FEFaceValues<dim> fe_face_values (fe, qf_face,
                                             update_values         | update_quadrature_points  |
                                             update_normal_vectors | update_JxW_values);
           double resultant_force = 0.0;
           double face_area = 0.0;

           typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active(),
           endc = dof_handler.end();
           for (; cell != endc; ++cell)
             if (cell->is_locally_owned())
           {

               PointHistory<dim> *lqph =
                                         reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)

                  if (cell->face(face)->at_boundary()
                     &&
                     (cell->face(face)->boundary_id() == 3))
                   {
                     fe_face_values.reinit (cell, face);
                     for (unsigned int q_point=0; q_point<n_q_points_f; ++q_point)
                       {
                         const Tensor<2, dim> F_inv_tr = lqph[q_point].get_F_inv_tr();
                         const Tensor<2, dim> tau      = lqph[q_point].get_tau();
                         const double J = lqph[q_point].get_det_F();

                         const Tensor<1, dim> element_force
                                 = (tau*F_inv_tr)* fe_face_values.normal_vector(q_point);

                         const double element_area = J * ((F_inv_tr*fe_face_values.normal_vector(q_point)).norm());

                         resultant_force += element_force [1]*fe_face_values.JxW(q_point);
                           face_area += element_area* fe_face_values.JxW(q_point);

                       }

                   }
           }

             resultant_stress[time.get_timestep()]= -1*Utilities::MPI::sum(resultant_force, mpi_communicator )/
                                                 Utilities::MPI::sum(face_area, mpi_communicator );


            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {

              std::ofstream myfile;
              myfile.open ("resultant_stress.txt");
             for(unsigned int n=0; n<time.get_timestep(); n++)
               myfile<< resultant_stress[n]<<std::endl;
             myfile.close();

            if (apply_strain==true)
             {

               static_stress[time.get_strainstep()]= resultant_stress[time.get_timestep()-1];
               std::ofstream myfile2;
               myfile2.open ("static_stress.txt");
               for(unsigned int n=0; n<time.get_strainstep(); n++)
                  myfile2<< static_stress[n]<<std::endl;
                  myfile2.close();
               time.strainincrement();
            }

            }
    }


  ///////////////////////////////////

// Newoton-Raphson nonlinear iterations for Mechanics problem
      template <int dim>
      void Solid<dim>::solve_nonlinear_timestep()
       {
         double initial_rhs_norm = 0.;
         unsigned int newton_iteration = 0;
         TrilinosWrappers::MPI::Vector  temp_solution_update(locally_owned_dofs, mpi_communicator);
         TrilinosWrappers::MPI::Vector  tmp(locally_owned_dofs, mpi_communicator);
         tmp = solution;
           for (; newton_iteration < 100;   ++newton_iteration)
             {
               make_constraints(newton_iteration);
               assemble_system ();


                 if (newton_iteration == 0){
                 initial_rhs_norm = system_rhs.l2_norm();
                 pcout << " Solving for Displacement:   " <<std::endl;
              }

              pcout<<"right hand side norm : "<<system_rhs.l2_norm();


                 const unsigned int n_iterations
                     = solve ();


                 temp_solution_update = solution_update;


                 tmp += temp_solution_update;
                 solution = tmp;

                 update_qph_incremental();

               pcout<<"   Number of CG iterations:  "<< n_iterations<<std::endl;

                if (newton_iteration > 0 && system_rhs.l2_norm() <= 1e-5 * initial_rhs_norm)
                 {
                  pcout << "CONVERGED! " << std::endl;
                  break;
                 }
               AssertThrow (newton_iteration < 99,
               ExcMessage("No convergence in nonlinear solver!"));

            }
       }

      /////////////////////////////////////////
// // Newoton-Raphson nonlinear iterations for Quinzburg-Landau
      template <int dim>
        void Solid<dim>::solve_nonlinear_timestep_eta()
         {
          old_solution_eta = solution_eta;
            double initial_rhs_norm = 0.;
            double previous_rhs_norm = 0.;
            double step_corrector = 1.0;
          unsigned int newton_iteration = 0;
          unsigned int n_iterations= 0;
          PETScWrappers::MPI::Vector  temp_solution_update_eta(locally_owned_dofs_eta, mpi_communicator);
          PETScWrappers::MPI::Vector  tmp_eta(locally_owned_dofs_eta, mpi_communicator);
          tmp_eta= solution_eta;


             for (; newton_iteration < 100;   ++newton_iteration)
               {
                   assemble_system_eta ();

                   if (newton_iteration == 0)
                      pcout << " Solving for Order Parameter:   "  <<std::endl;
                  else
                      pcout << '+' << n_iterations;

                   if (newton_iteration == 0)
                      initial_rhs_norm = system_rhs_eta.l2_norm();
                   else
                   {
                   step_corrector = abs(previous_rhs_norm-system_rhs_eta.l2_norm())/previous_rhs_norm;
                    if (step_corrector>1)
                    step_corrector=0.01;
                   }

                  previous_rhs_norm = system_rhs_eta.l2_norm();

                  n_iterations = solve_eta ();

                  temp_solution_update_eta = solution_update_eta;

                  tmp_eta.add(step_corrector,temp_solution_update_eta);

                  constraints_eta.distribute(tmp_eta);

                  solution_eta= tmp_eta;

                  update_qph_incremental();


                  if (newton_iteration > 0 && system_rhs_eta.l2_norm() <= 1e-6 * initial_rhs_norm)
                   {
                    pcout << " CG iterations per nonlinear step.     CONVERGED! " << std::endl;
                    break;
                   }
                 AssertThrow (newton_iteration < 99,
                 ExcMessage("No convergence in nonlinear solver!"));

              }
         }
  //////////////////////////////////////////////////
      template <int dim>
      void Solid<dim>::run()
      {
        make_grid(); // generates the geometry and mesh
        system_setup(); // sets up the system matrices

        PETScWrappers::MPI::Vector  tmp_solution_eta(locally_owned_dofs_eta, mpi_communicator);
        VectorTools::interpolate(dof_handler_eta, InitialValues<dim>(0), tmp_solution_eta); //initial eta
        solution_eta= tmp_solution_eta;
        old_solution_eta = solution_eta;

        update_qph_incremental();

    // this is the zeroth iteration for compute the initial distorted solution
    //of the body due to arbitrary distribution of initial eta

        solve_nonlinear_timestep();
        output_results();
        output_resultant_stress();

        time.increment();

     // computed actual time integration to update displacement and eta
        while (time.current() <= time.end())
          {
            //       if ( time.get_timestep()==2216)
            //             {
            //              VectorTools::interpolate(dof_handler_eta, InitialValues<dim>(1660), solution_eta); //initial eta
            //              old_solution_eta = solution_eta;
            //              update_qph_incremental();
            //             }

            pcout << std::endl
                  << "Time step #" << time.get_timestep() << "; "
                  << "advancing to t = " << time.current() << "."
                  << std::endl;
         solve_nonlinear_timestep_eta();
         solve_nonlinear_timestep();

               if ( time.get_timestep()>230 && load_step<12 )
                {
                 ++load_step;
                 pcout << " Load Step = " << load_step<< std::endl;
               }

         output_results();
         output_resultant_stress();
         time.increment();

          }
       }
}
/////////////////////////////////////////////////////////
  int main (int argc, char *argv[])
  {
    try
      {
        using namespace dealii;
        using namespace PhaseField;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        Solid<3> solid_3d("parameters.prm");
        solid_3d.run();
      }
    catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
      }
    catch (...)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
      }

    return 0;
  }

