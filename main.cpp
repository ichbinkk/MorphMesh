// Because of Mosek complications, we don't use static library if Mosek is used.
#ifdef LIBIGL_WITH_MOSEK
#ifdef IGL_STATIC_LIBRARY
#undef IGL_STATIC_LIBRARY
#endif
#endif

#include <igl/boundary_conditions.h>
#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/jet.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/normalize_row_sums.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/readTGF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/bbw.h>
//#include <igl/embree/bone_heat.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>

#define TUTORIAL_SHARED_PATH "../data"

typedef
  std::vector<Eigen::Quaterniond,Eigen::aligned_allocator<Eigen::Quaterniond> >
  RotationList;

//const Eigen::RowVector3d sea_green(70./255.,252./255.,167./255.);
const Eigen::RowVector3d sea_green(0,1,0);
int selected = 0;
Eigen::MatrixXd V,W,U,C,M;
Eigen::MatrixXi T,F,BE;
Eigen::VectorXi P;
RotationList pose;
double anim_t = 1.0;
double anim_t_dir = -0.03;

bool pre_draw(igl::opengl::glfw::Viewer & viewer)
{
  using namespace Eigen;
  using namespace std;
  if(viewer.core().is_animating)
  {
    // Interpolate pose and identity
    RotationList anim_pose(pose.size());
    for(int e = 0;e<pose.size();e++)
    {
      anim_pose[e] = pose[e].slerp(anim_t,Quaterniond::Identity());
    }
    // Propagate relative rotations via FK to retrieve absolute transformations
    RotationList vQ;
    vector<Vector3d> vT;
    igl::forward_kinematics(C,BE,P,anim_pose,vQ,vT);
    const int dim = C.cols();
    MatrixXd T(BE.rows()*(dim+1),dim);
    for(int e = 0;e<BE.rows();e++)
    {
      Affine3d a = Affine3d::Identity();
      a.translate(vT[e]);
      a.rotate(vQ[e]);
      T.block(e*(dim+1),0,dim+1,dim) =
        a.matrix().transpose().block(0,0,dim+1,dim);
    }
    // Compute deformation via LBS as matrix multiplication
    U = M*T;

    // Also deform skeleton edges
    MatrixXd CT;
    MatrixXi BET;
    igl::deform_skeleton(C,BE,T,CT,BET);

    viewer.data().set_vertices(U);
	
	//Change existing points color
	//viewer.data().set_points(U.row(468), RowVector3d(1, 0, 0));
	//viewer.data().point_size = 4;
	//MatrixXd fp(5, 3);
	//fp.row(0) = U.row(478);
	//fp.row(1) = U.row(2388);
	//fp.row(2) = U.row(2750);
	//fp.row(3) = U.row(3124);
	//fp.row(4) = U.row(3430);
	//viewer.data().set_points(fp, RowVector3d(1, 0, 0));

    //viewer.data().set_edges(CT,BET,sea_green);
    viewer.data().compute_normals();
    anim_t += anim_t_dir;
    anim_t_dir *= (anim_t>=1.0 || anim_t<=0.0?-1.0:1.0);
  }
  return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
  switch(key)
  {
    case ' ':
      viewer.core().is_animating = !viewer.core().is_animating;
      break;
    case '.':
      selected++;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      viewer.data().set_data(W.col(selected));
      break;
    case ',':
      selected--;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      viewer.data().set_data(W.col(selected));
      break;
  }
  return true;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;
  igl::readMESH(TUTORIAL_SHARED_PATH "/hand.mesh",V,T,F);
  U=V;
  igl::readTGF(TUTORIAL_SHARED_PATH "/hand.tgf",C,BE);
  // retrieve parents for forward kinematics
  igl::directed_edge_parents(BE,P);

  // Read pose as matrix of quaternions per row
  MatrixXd Q;
  igl::readDMAT(TUTORIAL_SHARED_PATH "/hand.dmat",Q);
  igl::column_to_quats(Q,pose);
  assert(pose.size() == BE.rows());

  // List of boundary indices (aka fixed value indices into VV)
  VectorXi b;
  // List of boundary conditions of each weight function
  MatrixXd bc;
  igl::boundary_conditions(V,T,C,VectorXi(),BE,MatrixXi(),b,bc);

  // compute BBW weights matrix
  igl::BBWData bbw_data;
  // only a few iterations for sake of demo
  bbw_data.active_set_params.max_iter = 8;
  bbw_data.verbosity = 2;
  if(!igl::bbw(V,T,b,bc,bbw_data,W))
  {
    return EXIT_FAILURE;
  }

  //MatrixXd Vsurf = V.topLeftCorner(F.maxCoeff()+1,V.cols());
  //MatrixXd Wsurf;
  //if(!igl::bone_heat(Vsurf,F,C,VectorXi(),BE,MatrixXi(),Wsurf))
  //{
  //  return false;
  //}
  //W.setConstant(V.rows(),Wsurf.cols(),1);
  //W.topLeftCorner(Wsurf.rows(),Wsurf.cols()) = Wsurf = Wsurf = Wsurf = Wsurf;

  // Normalize weights to sum to one
  igl::normalize_row_sums(W,W);
  // precompute linear blend skinning matrix
  igl::lbs_matrix(V,W,M);

  ////get points data
  //MatrixXd fp(5, 3);
  //fp.row(0) = U.row(478);
  //fp.row(1) = U.row(2388);
  //fp.row(2) = U.row(2750);
  //fp.row(3) = U.row(3124);
  //fp.row(4) = U.row(3430);
  ////get faces data
  //MatrixXd sp(3, 3);
  //sp.row(0) = U.row(2388);
  //sp.row(1) = U.row(2386);
  //sp.row(2) = U.row(592);
  MatrixXi sf(1, 3);
  sf << 2388, 2386, 592;
  
  //find neigbor face list
  int a[5] = { 478,2388,2750,3124,3430};
  vector<int> bb(a, a + 5);
  vector<vector<int> > A;
  vector<vector<int> > Ai;
  igl::adjacency_list(F, A);
  igl::vertex_triangle_adjacency(U,F,A,Ai);
  for (int v = 0; v < bb.size(); ++v) {
	  vector<int> f_list(A[bb[v]]);
	  //输出 A 全部元素
	  for (int i = 0; i < f_list.size(); i++) {
		  cout << f_list.at(i) << " ";		//如果访问越界，会抛出异常，比下标运算更安全，流畅
	  }
	  cout << endl;
	  vector<int> v_list(Ai[bb[v]]);
	  //输出 Ai 全部元素
	  for (int i = 0; i < v_list.size(); i++) {
		  cout << v_list.at(i) << " ";		//如果访问越界，会抛出异常，比下标运算更安全，流畅
	  }
	  cout << endl;
  }

  // Plot the mesh with pseudocolors
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(U, F);
  //viewer.data().set_colors(RowVector3d(0, 1, 0));
  
  /*set points and edges color*/
  //viewer.data().set_points(fp, RowVector3d(1, 0, 0));
  //viewer.data().point_size = 5;
  //viewer.data().set_edges(sp, se, RowVector3d(1, 0, 0));
  //viewer.append_mesh();
  //viewer.data().set_mesh(U, sf);
  //viewer.data().set_colors(RowVector3d(1, 0, 0));
  
  //设置面颜色
  MatrixXd sC(F.rows(), 3);
  sC << RowVector3d(0, 1, 0).replicate(F.rows(), 1);
  //sC.block<8, 3>(6984, 0) << RowVector3d(1, 0, 0).replicate(8, 1);
  for (int v = 0; v < bb.size(); ++v) {
	  vector<int> f_list(A[bb[v]]);
	  for (int i = 0; i < f_list.size(); i++) {
		  sC.block<1, 3>(f_list[i], 0) << RowVector3d(1, 0, 0);
	  }
  }
  //添加额外face的颜色
  int aa[10] = {7733,7744,6984,6989,5968,6037,5071,4883,4246,4223};
  //vector<int> bb(aa, aa + 10);
  bb.clear();
  bb.assign(aa, aa + 10);
  for (int i = 0; i < bb.size(); i++) {
	  sC.block<1, 3>(bb[i], 0) << RowVector3d(1, 0, 0);
  }
  //设置点颜色
 // MatrixXd sC(U.rows(), 3);
 // for (int v = 0; v < bb.size(); ++v) {
	//vector<int> v_list(A[bb[v]]);
	//for (int i = 0; i < v_list.size(); i++) {
	//	sC.row(v_list[i]) << RowVector3d(1, 0, 0);
	//}
  //}
  viewer.data().set_colors(sC);

  //viewer.data().set_data(W.col(selected));
  //viewer.data().set_edges(C,BE,sea_green);
  viewer.data().show_lines = false;
  viewer.data().show_overlay_depth = false;
  viewer.data().line_width = 1;
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core().is_animating = false;
  viewer.core().animation_max_fps = 30.;
  //set background color
  viewer.core().background_color.setOnes();
  //set face color
  //viewer.data().set_colors(RowVector3d(0,1,0));
   
  // Attach a menu plugin
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  // Customize the menu
  double doubleVariable = 0.1f; // Shared between two menus

  // Add content to the default menu window
  menu.callback_draw_viewer_menu = [&]()
  {
	  // Draw parent menu content
	  menu.draw_viewer_menu();

	  // Add new group
	  if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
	  {
		  // Expose variable directly ...
		  ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

		  // ... or using a custom callback
		  static bool boolVariable = true;
		  if (ImGui::Checkbox("bool", &boolVariable))
		  {
			  // do something
			  std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
		  }

		  // Expose an enumeration type
		  enum Orientation { Up = 0, Down, Left, Right };
		  static Orientation dir = Up;
		  ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

		  // We can also use a std::vector<std::string> defined dynamically
		  static int num_choices = 3;
		  static std::vector<std::string> choices;
		  static int idx_choice = 0;
		  if (ImGui::InputInt("Num letters", &num_choices))
		  {
			  num_choices = std::max(1, std::min(26, num_choices));
		  }
		  if (num_choices != (int)choices.size())
		  {
			  choices.resize(num_choices);
			  for (int i = 0; i < num_choices; ++i)
				  choices[i] = std::string(1, 'A' + i);
			  if (idx_choice >= num_choices)
				  idx_choice = num_choices - 1;
		  }
		  ImGui::Combo("Letter", &idx_choice, choices);

		  // Add a button
		  if (ImGui::Button("Print Hello", ImVec2(-1, 0)))
		  {
			  std::cout << "Hello\n";
		  }
	  }
  };

  cout<<
    "Press '.' to show next weight function."<<endl<<
    "Press ',' to show previous weight function."<<endl<<
    "Press [space] to toggle animation."<<endl;
  viewer.launch();
  return EXIT_SUCCESS;
}
