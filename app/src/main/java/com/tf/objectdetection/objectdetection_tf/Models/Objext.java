package com.tf.objectdetection.objectdetection_tf.Models;

public class Objext {

    private String title;
    private int num;

    public Objext(String title, int num) {
        this.title = title;
        this.num = num;
    }

    public void increment(){
        this.num+=1;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }

    @Override
    public String toString() {
        if(num>1){
            return num+" "+title+"s,";
        }else{
            return num+" "+title+",";
        }

    }
}
