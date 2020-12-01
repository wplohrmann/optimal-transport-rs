use ndarray::{Array, Array1, Array2};
use mcmf::{GraphBuilder, Vertex, Cost, Capacity};

pub fn calculate_1D_ot(a: &Array1::<i32>, b: &Array1::<i32>, cost: &Array2::<i32>) -> (i32, Array2::<u32>) {
    let mut graph = GraphBuilder::new();

    for (i, a_value) in a.iter().enumerate() {
        let a_node = ("a", i);
        graph.add_edge(Vertex::Source, a_node, Capacity(*a_value), Cost(0));
        for (j, b_value) in b.iter().enumerate() {
            let b_node = ("b", j);
            if i == 0 {
                graph.add_edge(b_node, Vertex::Sink, Capacity(*b_value), Cost(0));
            }
            graph.add_edge(a_node, b_node, Capacity(*b_value), Cost(cost[[i, j]]));
        }
    }

    let (flow_cost, paths) = graph.mcmf();

    let mut transport_plan: Array2::<u32> = Array::zeros(cost.raw_dim());
    for path in paths.iter() {
        let a_idx = path.vertices()[1].as_option().unwrap().1;
        let b_idx = path.vertices()[2].as_option().unwrap().1;
        transport_plan[[a_idx, b_idx]] = path.amount();
    }

    (flow_cost, transport_plan)
}
