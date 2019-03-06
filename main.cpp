#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/copyleft/marching_cubes.h>
#include <math.h>
#include <list>
#include <iostream>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;
Eigen::MatrixXd Pp;
Eigen::MatrixXd Pn;

Eigen::MatrixXd B;
Eigen::MatrixXd Bt;
Eigen::MatrixXd W;
Eigen::MatrixXd A;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;


// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
unsigned int polyDegree = 0;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;

// Parameter: grid resolution
unsigned int resolution = 20;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

double epsilon;
std::list<int> datast[10][10][10];
double xlen,ylen,zlen;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

//#################################################################################################

float seteps(Eigen::Vector3d m, Eigen::Vector3d M){
    epsilon = (m(0)-M(0))*(m(0)-M(0)) + (m(1)-M(1))*(m(1)-M(1)) + (m(2)-M(2))*(m(2)-M(2));
    epsilon = sqrt(epsilon);
    return 0.005*epsilon;
}

void initdatast(){
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = constrained_points.colwise().minCoeff();
    bb_max = constrained_points.colwise().maxCoeff();
    Eigen::RowVector3d dim = bb_max - bb_min;
    
    // Grid spacing
    const double dx = dim[0] / (double)(10);
    const double dy = dim[1] / (double)(10);
    const double dz = dim[2] / (double)(10);
    
    for(int i=0;i<constrained_points.rows();i++){
        int bx= (constrained_points(i,0)-bb_min(0))/dx;
        int by= (constrained_points(i,1)-bb_min(1))/dy;
        int bz= (constrained_points(i,2)-bb_min(2))/dz;
        
        bx= bx>9?9:bx;
        by= by>9?9:by;
        bz= bz>9?9:bz;
        
        datast[bx][by][bz].push_back(i);
        
    }
    
}

double dist(double x, double y, double z, int i){
    return sqrt( (x-constrained_points(i,0))*(x-constrained_points(i,0)) + (y-constrained_points(i,1))*(y-constrained_points(i,1)) + (z-constrained_points(i,2))*(z-constrained_points(i,2))  );
}

int closest(double x, double y, double z, std::list<int> out){
    double mindist=1000;
    int minpt;
    
    if(out.empty()){
        return -1;
    }
    
    std::list<int>::const_iterator iterator;
    for (iterator = out.begin(); iterator != out.end(); ++iterator) {
        if( dist(x,y,z,*iterator) < mindist){
            mindist=dist(x,y,z,*iterator);
            minpt=*iterator;
        }
    }
    return minpt;
    
}

std::list<int> within_h(double h, double x, double y, double z){
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = constrained_points.colwise().minCoeff();
    bb_max = constrained_points.colwise().maxCoeff();
    Eigen::RowVector3d dim = bb_max - bb_min;
    
    // datast spacing
    const double dx = dim[0] / (double)(10);
    const double dy = dim[1] / (double)(10);
    const double dz = dim[2] / (double)(10);
    
    int bx =x/dx;
    int by =y/dy;
    int bz =z/dz;
    
    std::list<int> out;
    
    for(int i= (x-bb_min(0)-h)/dx; i<= (x-bb_min(0)-0.001+h)/dx ; i++){
        if(i<0 || i>9){
            continue;
        }
        for(int j= (y-bb_min(1)-h)/dy; j<= (y-bb_min(1)-0.001+h)/dy ; j++){
            if(j<0 || j>9){
                continue;
            }
            for(int k= (z-bb_min(2)-h)/dz; k<= (z-bb_min(2)-0.001+h)/dz ; k++){
                if(k<0 || k>9){
                    continue;
                }
                out.insert(out.end(), datast[i][j][k].begin(), datast[i][j][k].end());
            }
            
        }
    }
    
    return out;
}

double wendwt(double r){
    if(r>=wendlandRadius){
        return 0;
    }
    double h = wendlandRadius;
    return ( pow((1- r/h),4) * (4*r/h + 1) );
}

void setW(double x, double y, double z){
    int no=constrained_points.rows();
    W.resize(no,no);
    for(int i=0;i<no;i++){
        W(i,i)= wendwt(dist(x,y,z,i));
    }
}
void initB1(){
    int no= constrained_points.rows();
    B.resize(no,4);
    for(int i=0;i<no;i++){
        double x = constrained_points(i,0);
        double y = constrained_points(i,1);
        double z = constrained_points(i,2);
        B(i,0)=1;
        B(i,1)=x;
        B(i,2)=y;
        B(i,3)=z;
    }
}

void initB2(){
    int no=constrained_points.rows();
    
    B.resize(no,10);
    for(int i=0;i<no;i++){
        double x = constrained_points(i,0);
        double y = constrained_points(i,1);
        double z = constrained_points(i,2);
        
        B(i,0)=1;
        B(i,1)=x;
        B(i,2)=y;
        B(i,3)=z;
        B(i,4)=x*x;
        B(i,5)=y*y;
        B(i,6)=z*z;
        B(i,7)=x*y;
        B(i,8)=x*z;
        B(i,9)=y*z;
    }
}

void setA(double x, double y, double z){
    setW(x, y, z);
    Eigen::MatrixXd I = Bt * W * B;
    A = I.inverse() * (Bt * W * constrained_values);
    
}

double fvalue(double x, double y, double z){
    setA(x, y, z);
    double f =A(0);
    f+=A(1)*x + A(2)*y + A(3)*z;
    f+=A(4)*x*x + A(5)*y*y + A(6)*z*z;
    f+=A(7)*x*z + A(8)*x*y + A(9)*y*z;
    return f;
}

//#################################################################################################

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid() {
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines. resize(0, 6);
    grid_values.resize(0);
    V. resize(0, 3);
    F. resize(0, 3);
    FN.resize(0, 3);
    
    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = constrained_points.colwise().minCoeff();
    bb_max = constrained_points.colwise().maxCoeff();
    
    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;
    
    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
void evaluateImplicitFunc() {
    
    // Sphere center
    auto bb_min = grid_points.colwise().minCoeff().eval();
    auto bb_max = grid_points.colwise().maxCoeff().eval();
    Eigen::RowVector3d center = 0.5 * (bb_min + bb_max);
    
    Eigen::RowVector3d dim = bb_max - bb_min;
    
    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);
    
    // Evaluate sphere's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                double bx = bb_min(0) + x*dim[0]/(resolution -1);
                double by = bb_min(1) + x*dim[1]/(resolution -1);
                double bz = bb_min(2) + x*dim[2]/(resolution -1);
                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = fvalue(bx,by,bz);
                cout<<"  "<<x<<"  "<<y<<"  "<<z<<"  ";
            }
        }
    }
}



// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;
    
    for (unsigned int x = 0; x<resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }
    
    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Show imported points
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        viewer.data().point_size = 11;
        viewer.data().add_points(P, Eigen::RowVector3d(0,0,0));
    }
    if (key == '0') {
        
        
        //cout<<" ******** "<<P(0,0)<<" ********** "<<P(0,1)<<" ***********  "<<N(0,0)<<" ***********  "<<N(0,1);
    }
    
    
    if (key == '2') {
        int no = P.rows();
        // Show all constraints
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        
        Eigen::Vector3d m = P.colwise().minCoeff();
        Eigen::Vector3d M = P.colwise().maxCoeff();
        epsilon=seteps(m,M);
        
        Pp.resize(P.rows(),P.cols());
        Pn.resize(P.rows(),P.cols());
        
        // Add your code for computing auxiliary constraint points here
        for(int i=0;i<P.rows();i++){
            for(int j=0;j<P.cols();j++){
                Pp(i,j)=P(i,j)+epsilon*N(i,j);
                Pn(i,j)=P(i,j)-epsilon*N(i,j);
            }
        }
        
        constrained_points.resize(3*P.rows(),P.cols());
        constrained_points << P, Pp, Pn;
        
        int bigrows=constrained_points.rows();
        constrained_values.resize(bigrows,1);
        
        for(int i=0;i<bigrows;i++){
            if(i<P.rows()){
                constrained_values(i)=0;
            }
            else if(i<2*P.rows()){
                constrained_values(i)= epsilon;
            }
            else{
                constrained_values(i)= -epsilon;
            }
            
        }
        /*
         for(int i=P.cols();i<2*P.cols();i++){
         if( closestpt(constrained_points(i,0),constrained_points(i,1),constrained_points(i,2)) &&
         closestpt(constrained_points(i+no,0),constrained_points(i+no,1),constrained_points(i+no,2))
         ){
         
         }
         if( dist(P(i-no,0),P(i-no,1),P(i-no,2),P(i,0),P(i,1),P(i,2)) < )
         }
         */
        
        // Add code for displaying all points, as above
        viewer.core.align_camera_center(P);
        viewer.data().point_size = 5;
        viewer.data().add_points(P, Eigen::RowVector3d(0,0,1));
        viewer.data().add_points(Pp, Eigen::RowVector3d(0,1,0));
        viewer.data().add_points(Pn, Eigen::RowVector3d(1,0,0));
        
        initdatast();
    }
    
    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        // Add code for creating a grid
        // Add your code for evaluating the implicit function at the grid points
        // Add code for displaying points and lines
        // You can use the following example:
        
        
        initB2();
        Bt=B.transpose();
        //cout<<"    fval   ************    "<<fvalue(P(0,0),P(0,1),P(0,2));
        
        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();
        
        // Evaluate implicit function
        evaluateImplicitFunc();
        
        // get grid lines
        getLines();
        
        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);
        
        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }
        
        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                                grid_lines.block(0, 3, grid_lines.rows(), 3),
                                Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
    }
    
    if (key == '4') {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }
        
        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
    }
    
    return true;
}



int main(int argc, char *argv[]) {
    //if (argc != 2) {
    //  cout << argc << endl;
    //exit(0);
    //}
    
    // Read points and normals
    igl::readOFF("bunny-500.off",P,F,N);
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    
    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    
    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();
        // Add widgets to the sidebar.
        if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputScalar("resolution", ImGuiDataType_U32, &resolution);
            if (ImGui::Button("Reset Grid", ImVec2(-1,0))) {
                // Recreate the grid
                createGrid();
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }
            
            // TODO: Add more parameters to tweak here...
            ImGui::InputScalar("wendland weight", ImGuiDataType_Double, &wendlandRadius);
            if (ImGui::Button("Reset Grid", ImVec2(-1,0))) {
                // Recreate the grid
                createGrid();
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }
            
            
            ImGui::InputScalar("polyDegree", ImGuiDataType_U32, &polyDegree);
            if (ImGui::Button("Reset Grid", ImVec2(-1,0))) {
                // Recreate the grid
                createGrid();
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }
            
            
            
        }
        
    };
    
    viewer.launch();
}
