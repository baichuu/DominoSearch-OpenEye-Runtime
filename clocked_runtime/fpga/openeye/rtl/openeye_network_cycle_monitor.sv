// Observe one uninterrupted network transaction on a single clock.
// start_* describes ONLY the first input beat; finish_* ONLY the final output.
// Occupancy counts include both boundary edges. elapsed_cycles_o is the edge
// distance used for latency: a same-edge transaction has zero-cycle latency.
// Reset invalidates the previous measurement, including any unfinished run.
// This observer must be wired to real graph handshakes before reporting latency.
module openeye_network_cycle_monitor #(
    parameter integer COUNTER_WIDTH = 64
) (
    input wire clk_i,
    input wire rst_ni,
    input wire start_valid_i,
    input wire start_ready_i,
    input wire finish_valid_i,
    input wire finish_ready_i,
    // Diagnostic occupancy bits may overlap. They never add to total_cycles_o.
    input wire compute_active_i,
    input wire postops_active_i,
    input wire transfer_active_i,
    output reg running_o,
    output reg done_o,
    output reg invalid_o,
    output wire [COUNTER_WIDTH-1:0] elapsed_cycles_o,
    // Inclusive observation slots, NOT the edge distance for latency.
    output reg [COUNTER_WIDTH-1:0] total_cycles_o,
    output reg [COUNTER_WIDTH-1:0] compute_cycles_o,
    output reg [COUNTER_WIDTH-1:0] postops_cycles_o,
    output reg [COUNTER_WIDTH-1:0] transfer_cycles_o,
    output reg [COUNTER_WIDTH-1:0] idle_cycles_o,
    output reg [COUNTER_WIDTH-1:0] overlap_cycles_o
);
    wire start_fire = start_valid_i && start_ready_i;
    wire finish_fire = finish_valid_i && finish_ready_i;
    wire any_active = compute_active_i || postops_active_i || transfer_active_i;
    wire overlap = (compute_active_i && postops_active_i) ||
                   (compute_active_i && transfer_active_i) ||
                   (postops_active_i && transfer_active_i);
    assign elapsed_cycles_o = total_cycles_o == 0 ? 0 : total_cycles_o - 1'b1;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            running_o <= 1'b0;
            done_o <= 1'b0;
            invalid_o <= 1'b0;
            total_cycles_o <= 0;
            compute_cycles_o <= 0;
            postops_cycles_o <= 0;
            transfer_cycles_o <= 0;
            idle_cycles_o <= 0;
            overlap_cycles_o <= 0;
        end else begin
            // One measurement per reset; duplicate or reversed boundaries fail.
            if ((start_fire && (running_o || done_o)) ||
                (finish_fire && ((!running_o && !start_fire) || done_o)))
                invalid_o <= 1'b1;

            if (!done_o && (running_o || start_fire)) begin
                if (&total_cycles_o) begin
                    // Saturate instead of silently wrapping a long run to zero.
                    invalid_o <= 1'b1;
                end else begin
                    total_cycles_o <= total_cycles_o + 1'b1;
                    compute_cycles_o <= compute_cycles_o + compute_active_i;
                    postops_cycles_o <= postops_cycles_o + postops_active_i;
                    transfer_cycles_o <= transfer_cycles_o + transfer_active_i;
                    idle_cycles_o <= idle_cycles_o + !any_active;
                    overlap_cycles_o <= overlap_cycles_o + overlap;
                end
                running_o <= !finish_fire;
                if (finish_fire) done_o <= 1'b1;
            end
        end
    end
endmodule
