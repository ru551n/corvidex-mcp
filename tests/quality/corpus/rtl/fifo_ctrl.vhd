library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fifo_ctrl is
  -- Simple synchronous FIFO controller. The write pointer and the read
  -- pointer free-run; full and empty are derived from their distance.
  -- The asynchronous reset rst_n clears both pointers.
  generic (
    DEPTH : natural := 16
  );
  port (
    clk     : in  std_logic;
    rst_n   : in  std_logic;
    wr_en   : in  std_logic;
    rd_en   : in  std_logic;
    full    : out std_logic;
    empty   : out std_logic
  );
end entity fifo_ctrl;

architecture rtl of fifo_ctrl is
  type ptr_t is range 0 to DEPTH;
  signal wr_ptr : ptr_t := 0;
  signal rd_ptr : ptr_t := 0;
begin
  comb : process (wr_ptr, rd_ptr)
  begin
    full  <= '1' when wr_ptr - rd_ptr = DEPTH else '0';
    empty <= '1' when wr_ptr = rd_ptr else '0';
  end process comb;

  ptrs : process (clk, rst_n)
  begin
    if rst_n = '0' then
      wr_ptr <= 0;
      rd_ptr <= 0;
    elsif rising_edge(clk) then
      if wr_en = '1' and not (wr_ptr = rd_ptr and DEPTH = 1) then
        wr_ptr <= wr_ptr + 1;
      end if;
      if rd_en = '1' and wr_ptr /= rd_ptr then
        rd_ptr <= rd_ptr + 1;
      end if;
    end if;
  end process ptrs;
end architecture rtl;