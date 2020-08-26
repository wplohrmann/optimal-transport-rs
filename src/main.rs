use ndarray::{Array, Array1, Array2, array};
use mcmf::{GraphBuilder, Vertex, Cost, Capacity};

fn main() {
    let a = array![1, 2, 3];
    let b = array![2, 2, 2];
    let cost = array![[0, 0, 0], [0, 0, 0], [3, 3, 0]];
    let transport_plan = calculate_1D_ot(&a, &b, &cost);
    println!("from: {}", a);
    println!("to: {}", b);
    println!("transport plan:");
    println!("{:?}", transport_plan);
}

fn calculate_1D_ot(a: &Array1::<i32>, b: &Array1::<i32>, cost: &Array2::<i32>) -> Array2::<u32> {
    let mut graph = GraphBuilder::new();

    for (i, a_value) in a.iter().enumerate() {
        let a_node = ("a", i);
        graph.add_edge(Vertex::Source, a_node, Capacity(*a_value), Cost(0));
        for (j, b_value) in b.iter().enumerate() {
            let b_node = ("b", j);
            graph.add_edge(a_node, b_node, Capacity(*b_value), Cost(cost[[i, j]]));
        }
    }
    for (j, b_value) in b.iter().enumerate() {
        let b_node = ("b", j);
        graph.add_edge(b_node, Vertex::Sink, Capacity(*b_value), Cost(0));
    }

    let (flow_cost, paths) = graph.mcmf();
    // for path in paths.iter() {
    //     println!("{}", path.cost());
    // }
    // println!("flow_cost: {}, paths: {:?}", flow_cost, paths.len());

    let mut transport_plan: Array2::<u32> = Array::zeros(cost.raw_dim());
    for path in paths.iter() {
        let a_idx = path.vertices()[1].as_option().unwrap().1;
        let b_idx = path.vertices()[2].as_option().unwrap().1;
        transport_plan[[a_idx, b_idx]] = path.amount();
        // println!("{:?}, {:?}", path.vertices()[1].as_option().unwrap().1, path.vertices()[2]);
        // println!("{}", path.amount())
    }

    transport_plan
}
